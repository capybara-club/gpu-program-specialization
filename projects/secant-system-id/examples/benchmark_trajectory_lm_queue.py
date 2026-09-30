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
import json
import math
import statistics
import time

from secant_system_id.ast import Program, input_slot
from secant_system_id.c_runtime import C99Library, GPCandidate
from secant_system_id.compiler import compile_cuda
from secant_system_id.fed_batch import (
    FED_BATCH_SHAPE,
    PLANTED_BINDINGS,
    PLANTED_CONSTANTS,
    PRODUCT_PAIRED_INITIAL_STATES,
    planted_programs,
    reference_data,
)
from secant_system_id.genome import SystemGenome
from secant_system_id.lm_specialization import specialize_split_lm_cubin
from secant_system_id.trajectory_lm import (
    TrajectoryLMQueue,
    make_lm_population,
    render_trajectory_lm_source,
    run_trajectory_lm,
)


def _genomes(count: int) -> tuple[SystemGenome, ...]:
    planted = planted_programs()
    first_offset, second_offset = FED_BATCH_SHAPE.ast_input_offsets
    a = [input_slot(first_offset + index) for index in range(8)]
    b = [input_slot(second_offset + index) for index in range(8)]
    alternatives = (
        SystemGenome(planted),
        SystemGenome(
            (
                Program.from_expression(a[0] * a[1] / (a[2] + a[3] * a[4])),
                planted[1],
            )
        ),
        SystemGenome(
            (
                planted[0],
                Program.from_expression(b[0] * b[1] / (b[2] + b[3] * b[4])),
            )
        ),
        SystemGenome(
            (
                Program.from_expression((a[0] + a[1]) / (a[2] + a[3] * a[4])),
                Program.from_expression((b[0] + b[1]) / (b[2] + b[3] * b[4])),
            )
        ),
    )
    return tuple(alternatives[index % len(alternatives)] for index in range(count))


def _maximum_error(expected: array, actual: array) -> float:
    if len(expected) != len(actual):
        return math.inf
    maximum = 0.0
    for lhs, rhs in zip(expected, actual):
        if math.isfinite(lhs) and math.isfinite(rhs):
            maximum = max(maximum, abs(lhs - rhs))
        elif math.isnan(lhs) != math.isnan(rhs) or math.isinf(lhs) != math.isinf(rhs):
            return math.inf
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--nvcc")
    parser.add_argument("--c99-library")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--settings", type=int, default=8192)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--damping-attempts", type=int, default=8)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--loaded-modules", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--skip-direct", action="store_true")
    arguments = parser.parse_args()
    if min(
        arguments.candidates,
        arguments.settings,
        arguments.starts,
        arguments.iterations,
        arguments.damping_attempts,
        arguments.threads,
        arguments.workers,
        arguments.loaded_modules,
        arguments.repetitions,
    ) <= 0:
        raise ValueError("all benchmark dimensions must be positive")

    compilation = compile_cuda(
        render_trajectory_lm_source(), arguments.arch, arguments.nvcc
    )
    genomes = _genomes(arguments.candidates)
    specializations = tuple(
        specialize_split_lm_cubin(compilation.cubin, genome, FED_BATCH_SHAPE)[0]
        for genome in genomes
    )
    populations = tuple(
        make_lm_population(
            GPCandidate(
                index,
                1.0,
                1.0,
                0,
                sum(len(tuple(program.instructions())) for program in genome.programs),
                tuple(program.data for program in genome.programs),
                PLANTED_CONSTANTS,
                PLANTED_BINDINGS,
            ),
            arguments.settings,
            arguments.starts,
            1009 + index,
        )
        for index, genome in enumerate(genomes)
    )
    reference = reference_data(16, PRODUCT_PAIRED_INITIAL_STATES, 12)

    direct = ()
    direct_wall_seconds = None
    if not arguments.skip_direct:
        started = time.perf_counter()
        direct = tuple(
            run_trajectory_lm(
                specialization.cubin,
                population,
                reference,
                max_lm_iterations=arguments.iterations,
                max_damping_attempts=arguments.damping_attempts,
                threads_per_block=arguments.threads,
            )
            for specialization, population in zip(specializations, populations)
        )
        direct_wall_seconds = time.perf_counter() - started

    api = C99Library(arguments.c99_library)
    queued_runs = []
    with TrajectoryLMQueue(
        api,
        len(compilation.cubin),
        reference,
        arguments.candidates,
        arguments.settings,
        arguments.starts,
        arguments.workers,
        arguments.loaded_modules,
    ) as queue:
        for _ in range(arguments.repetitions):
            queued_runs.append(
                queue.run(
                    tuple(result.cubin for result in specializations),
                    populations,
                    max_lm_iterations=arguments.iterations,
                    max_damping_attempts=arguments.damping_attempts,
                    threads_per_block=arguments.threads,
                )
            )

    comparison = []
    if direct:
        for expected, actual in zip(direct, queued_runs[-1].results):
            comparison.append(
                {
                    "mse_max_abs_error": _maximum_error(expected.mse, actual.mse),
                    "constants_max_abs_error": _maximum_error(
                        expected.constants, actual.constants
                    ),
                    "iterations_equal": expected.iterations == actual.iterations,
                    "accepted_steps_equal": expected.accepted_steps == actual.accepted_steps,
                    "best_fit_equal": expected.best_fit == actual.best_fit,
                }
            )
    print(
        json.dumps(
            {
                "architecture": arguments.arch,
                "candidates": arguments.candidates,
                "fits_per_candidate": arguments.settings * arguments.starts,
                "total_fits": arguments.candidates * arguments.settings * arguments.starts,
                "compile_seconds": compilation.elapsed_seconds,
                "cubin_bytes": len(compilation.cubin),
                "register_counts": [result.register_count for result in specializations],
                "workers": arguments.workers,
                "maximum_loaded_modules": arguments.loaded_modules,
                "direct_wall_seconds": direct_wall_seconds,
                "queued_wall_seconds": [result.wall_seconds for result in queued_runs],
                "queued_median_wall_seconds": statistics.median(
                    result.wall_seconds for result in queued_runs
                ),
                "queued_fits_per_second": [
                    arguments.candidates
                    * arguments.settings
                    * arguments.starts
                    / result.wall_seconds
                    for result in queued_runs
                ],
                "module_load_seconds": [
                    list(result.module_load_seconds) for result in queued_runs
                ],
                "comparison": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
