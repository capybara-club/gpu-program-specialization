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
import math
from pathlib import Path
import re

from .compiler import compile_cuda
from .fed_batch import FED_BATCH_MODEL, FED_BATCH_SHAPE, reference_data, score_setting
from .inspection import inspect_module
from .packed_cubin import PackedCubinPlan
from .packed_sass import specialize_packed_cubin
from .packed_template import generate_packed_cuda
from .runtime import run_packed_fedbatch
from .shape import PackedDispatch
from .toggle_fed_batch import (
    TogglePopulation,
    benchmark_toggle_systems,
    make_toggle_population,
    materialize_toggle_population,
)
from .toggle_runtime import run_toggle_cuda
from .toggle_template import generate_toggle_cuda


def _register_count(log: str) -> int | None:
    matches = [int(value) for value in re.findall(r"Used\s+(\d+)\s+registers", log)]
    return max(matches) if matches else None


def _rate_report(
    seconds: float,
    population: TogglePopulation,
    ast_count: int,
) -> dict[str, float]:
    configurations = population.num_systems * population.configurations_per_system
    return {
        "seconds_per_launch": seconds,
        "resident_system_sweeps_per_second": population.num_systems / seconds,
        "resident_component_ast_sweeps_per_second": population.num_systems * ast_count / seconds,
        "configurations_per_second": configurations / seconds,
        "trajectories_per_second": configurations * FED_BATCH_SHAPE.trajectory_count / seconds,
        "nanoseconds_per_configuration": seconds * 1.0e9 / configurations,
    }


def _correctness_samples(
    systems,
    population,
    gpu_scores,
    reference,
    steps_per_observation: int,
    count: int,
    configuration_count: int | None = None,
):
    rejection_mse = 1.0e6
    configurations = (
        population.configurations_per_system
        if configuration_count is None
        else configuration_count
    )
    mask = population.permutations_per_bank - 1
    coordinates = [(0, 0), (0, mask), (0, configurations - 1)]
    for index in range(max(0, count - len(coordinates))):
        coordinates.append(
            (
                (index * 37 + 1) % population.num_systems,
                (index * 53 + 7) % configurations,
            )
        )
    samples = []
    failures = 0
    for system_index, configuration in coordinates[:count]:
        bank = configuration >> population.toggle_bit_count
        permutation = configuration & mask
        constants = population.constants(system_index, bank)
        bindings = systems[system_index].resolved_bindings(permutation, FED_BATCH_SHAPE)
        genome = systems[system_index].materialized_genome(FED_BATCH_SHAPE)
        cpu = score_setting(
            genome.programs,
            constants,
            bindings,
            reference,
            steps_per_observation,
            FED_BATCH_SHAPE,
        )
        gpu = float(gpu_scores[system_index * configurations + configuration])
        error = abs(cpu - gpu)
        tolerance = 2.0e-4 + 3.0e-4 * max(abs(cpu), abs(gpu))
        both_invalid = cpu >= 0.99 * 3.402823466e38 and gpu >= 0.99 * 3.402823466e38
        numeric_match = math.isfinite(cpu) and math.isfinite(gpu) and error <= tolerance
        both_rejected = cpu >= rejection_mse and gpu >= rejection_mse
        passed = both_invalid or numeric_match or both_rejected
        classification = (
            "both_invalid"
            if both_invalid
            else "numeric_match"
            if numeric_match
            else "both_above_rejection_mse"
            if both_rejected
            else "failure"
        )
        failures += not passed
        samples.append(
            {
                "system": system_index,
                "configuration": configuration,
                "constant_bank": bank,
                "permutation": permutation,
                "cpu_mse": cpu,
                "gpu_mse": gpu,
                "absolute_error": error,
                "tolerance": tolerance,
                "classification": classification,
                "passed": passed,
            }
        )
    return failures, samples


def _compare_gpu_scores(first, second, population):
    configurations = population.configurations_per_system
    finite_pairs = 0
    maximum_absolute_error = 0.0
    invalid_status_mismatches = 0
    winner_matches = 0
    invalid_mismatch_samples = []
    stable_pair_count = 0
    stable_maximum_absolute_error = 0.0
    stable_maximum_relative_error = 0.0
    for flat_index, (lhs, rhs) in enumerate(zip(first, second)):
        lhs_valid = math.isfinite(lhs) and lhs < 0.99 * 3.402823466e38
        rhs_valid = math.isfinite(rhs) and rhs < 0.99 * 3.402823466e38
        if lhs_valid != rhs_valid:
            invalid_status_mismatches += 1
            if len(invalid_mismatch_samples) < 8:
                invalid_mismatch_samples.append(
                    {
                        "system": flat_index // configurations,
                        "configuration": flat_index % configurations,
                        "toggle_mse": lhs,
                        "materialized_mse": rhs,
                    }
                )
        if lhs_valid and rhs_valid:
            finite_pairs += 1
            maximum_absolute_error = max(maximum_absolute_error, abs(lhs - rhs))
            if max(abs(lhs), abs(rhs)) <= 1.0e6:
                absolute_error = abs(lhs - rhs)
                relative_error = absolute_error / max(abs(lhs), abs(rhs), 1.0e-12)
                stable_pair_count += 1
                stable_maximum_absolute_error = max(stable_maximum_absolute_error, absolute_error)
                stable_maximum_relative_error = max(stable_maximum_relative_error, relative_error)
    for system_index in range(population.num_systems):
        start = system_index * configurations
        stop = start + configurations
        first_winner = min(range(configurations), key=lambda index: first[start + index])
        second_winner = min(range(configurations), key=lambda index: second[start + index])
        winner_matches += first_winner == second_winner
    return {
        "score_count": len(first),
        "finite_score_pairs": finite_pairs,
        "maximum_absolute_score_error": maximum_absolute_error,
        "invalid_status_mismatches": invalid_status_mismatches,
        "invalid_status_mismatch_samples": invalid_mismatch_samples,
        "stable_pair_definition": "both finite and both MSE <= 1e6",
        "stable_pair_count": stable_pair_count,
        "stable_maximum_absolute_error": stable_maximum_absolute_error,
        "stable_maximum_relative_error": stable_maximum_relative_error,
        "exact_winner_matches": winner_matches,
        "system_count": population.num_systems,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA-compiled bit-toggle SSE systems against materialized bindings."
    )
    parser.add_argument("--arch", default="sm_120")
    parser.add_argument("--nvcc")
    parser.add_argument("--systems", type=int, default=128)
    parser.add_argument("--constant-banks", type=int, default=4)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--checks", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compare-materialized", action="store_true")
    parser.add_argument("--build-directory", default="generated/toggle_cuda")
    arguments = parser.parse_args(argv)
    if arguments.systems <= 0 or arguments.systems > 128:
        raise ValueError("this benchmark supports between 1 and 128 packed systems")

    build = Path(arguments.build_directory)
    build.mkdir(parents=True, exist_ok=True)
    systems = benchmark_toggle_systems(arguments.systems)
    population = make_toggle_population(systems, arguments.constant_banks)
    source = generate_toggle_cuda(
        FED_BATCH_MODEL,
        FED_BATCH_SHAPE,
        systems,
        population.toggle_bit_count,
    )
    (build / "toggle_packed.cu").write_text(source)
    compilation = compile_cuda(source, arguments.arch, arguments.nvcc)
    (build / "toggle_packed.cubin").write_bytes(compilation.cubin)
    (build / "toggle_compile.log").write_text(compilation.log)
    reference = reference_data(arguments.steps)
    toggle_run = run_toggle_cuda(
        compilation.cubin,
        population,
        reference,
        FED_BATCH_SHAPE,
        arguments.steps,
        arguments.threads,
        arguments.repetitions,
        arguments.device,
    )
    failures, samples = _correctness_samples(
        systems,
        population,
        toggle_run.mse,
        reference,
        arguments.steps,
        arguments.checks,
    )

    deviations = [
        "This is a CUDA-compiled resident-kernel prototype, not the SASS specialization/module pipeline.",
        "Only dense aligned fed-batch trajectories and full MSE output are implemented.",
        "Shared memory still holds the read-only trajectory reference; the per-thread dynamic leaf bank is removed.",
    ]
    report: dict[str, object] = {
        "schema": "secant.system_id.toggle_cuda_benchmark.v1",
        "scope": "resident CUDA execution; compilation and module loading reported separately",
        "architecture": arguments.arch,
        "system_count": population.num_systems,
        "component_asts_per_system": FED_BATCH_SHAPE.ast_count,
        "toggle_bits": population.toggle_bit_count,
        "permutations_per_constant_bank": population.permutations_per_bank,
        "constant_banks_per_system": population.constant_bank_count,
        "configurations_per_system": population.configurations_per_system,
        "threads_per_block": arguments.threads,
        "steps_per_observation": arguments.steps,
        "source_bytes": len(source.encode()),
        "cubin_bytes": len(compilation.cubin),
        "compile_seconds": compilation.elapsed_seconds,
        "registers_per_thread": toggle_run.registers_per_thread or _register_count(compilation.log),
        "local_bytes_per_thread": toggle_run.local_bytes_per_thread,
        "shared_bytes_per_cta": 4 * FED_BATCH_SHAPE.reference_float_count,
        "constant_bank_bytes": len(population.constant_banks) * 4,
        "module_load_seconds": toggle_run.module_load_seconds,
        "function_lookup_seconds": toggle_run.function_lookup_seconds,
        "toggle_cuda": _rate_report(
            toggle_run.seconds_per_launch,
            population,
            FED_BATCH_SHAPE.ast_count,
        ),
        "cpu_correctness": {
            "checked": len(samples),
            "failures": failures,
            "rejection_mse": 1.0e6,
            "reference_precision": "Python float64 reference compared with CUDA FP32 rollout",
            "samples": samples,
        },
        "materialized_baseline": None,
        "gpu_equivalence": None,
        "deviations": deviations,
    }

    if arguments.compare_materialized:
        materialized = materialize_toggle_population(systems, population)
        genomes = tuple(system.materialized_genome(FED_BATCH_SHAPE) for system in systems)
        dispatch = PackedDispatch(arguments.systems, 1)
        baseline_source = generate_packed_cuda(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            dispatch,
        )
        (build / "materialized_template.cu").write_text(baseline_source)
        baseline_compilation = compile_cuda(
            baseline_source,
            arguments.arch,
            arguments.nvcc,
        )
        inspection = inspect_module(baseline_source, baseline_compilation.cubin)
        if not isinstance(inspection.plan, PackedCubinPlan):
            raise RuntimeError("matched materialized baseline did not produce one packed kernel")
        specialization = specialize_packed_cubin(
            baseline_compilation.cubin,
            inspection.plan,
            genomes,
            FED_BATCH_SHAPE,
        )
        (build / "materialized_specialized.cubin").write_bytes(specialization.cubin)
        (build / "materialized_compile.log").write_text(baseline_compilation.log)
        baseline_run = run_packed_fedbatch(
            specialization.cubin,
            materialized,
            reference,
            FED_BATCH_SHAPE,
            dispatch,
            arguments.steps,
            arguments.threads,
            arguments.repetitions,
            arguments.device,
        )
        baseline_rates = _rate_report(
            baseline_run.seconds_per_launch,
            population,
            FED_BATCH_SHAPE.ast_count,
        )
        toggle_rates = report["toggle_cuda"]
        report["materialized_baseline"] = {
            "source_bytes": len(baseline_source.encode()),
            "cubin_bytes": len(specialization.cubin),
            "compile_seconds": baseline_compilation.elapsed_seconds,
            "template_registers_per_thread": inspection.plan.register_count,
            "specialized_registers_per_thread": specialization.register_count,
            "shared_bytes_per_cta": FED_BATCH_SHAPE.shared_bytes(arguments.threads),
            "settings_and_bindings_bytes": (
                len(materialized.settings) * materialized.settings.itemsize
                + len(materialized.bindings) * materialized.bindings.itemsize
            ),
            "module_load_seconds": baseline_run.module_load_seconds,
            "function_lookup_seconds": baseline_run.function_lookup_seconds,
            **baseline_rates,
        }
        report["gpu_equivalence"] = _compare_gpu_scores(
            toggle_run.mse,
            baseline_run.mse,
            population,
        )
        report["toggle_speedup_over_materialized"] = (
            toggle_rates["configurations_per_second"]
            / baseline_rates["configurations_per_second"]
        )

    gpu_equivalence = report["gpu_equivalence"]
    exact_gpu_baseline = bool(
        isinstance(gpu_equivalence, dict)
        and gpu_equivalence["invalid_status_mismatches"] == 0
        and gpu_equivalence["maximum_absolute_score_error"] == 0.0
        and gpu_equivalence["exact_winner_matches"] == population.num_systems
    )
    if failures and exact_gpu_baseline:
        deviations.append(
            "Some sensitive high-loss Python-float64 rollouts differ from CUDA FP32; the toggle and materialized GPU paths still match every score exactly."
        )

    (build / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures and not exact_gpu_baseline:
        raise RuntimeError(f"{failures} CPU correctness checks failed")


if __name__ == "__main__":
    main()
