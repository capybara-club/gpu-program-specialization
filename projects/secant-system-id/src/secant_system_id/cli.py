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
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

from .c_runtime import (
    C99Library,
    CFedbatchLaunch,
    CGPRunStats,
    FlatGenomeBatch,
    GPCheckpoint,
    GPConfig,
    SSID_OUTPUT_GENOME_WINNERS,
    SSID_SETTINGS_HASHED_INCUMBENT,
    SSID_SETTINGS_MATERIALIZED,
    benchmark_specialization,
)
from .ast import InstructionType, Program
from .autodiff import DerivativeTapeError
from .compiler import compile_cuda
from .cubin import inspect_cubin
from .fed_batch import (
    DIVERSE_INITIAL_STATES,
    FED_BATCH_MODEL,
    FED_BATCH_SHAPE,
    INITIAL_STATES,
    INVALID_MSE,
    PackedPopulation,
    PRODUCT_PAIRED_INITIAL_STATES,
    benchmark_genomes,
    make_population,
    pack_population,
    packed_setting_values,
    planted_programs,
    reference_data,
    score_packed_population,
    score_setting,
)
from .model import SystemModel
from .genome import SystemGenome
from .gpu_gradient_check import random_genome
from .inspection import inspect_module
from .packed_cubin import PackedModulePlan, inspect_packed_cubin
from .packed_sass import specialize_packed_cubin, specialize_packed_module_cubin
from .packed_template import generate_packed_cuda, generate_packed_cuda_module
from .runtime import run_fedbatch, run_packed_fedbatch, run_packed_module_fedbatch
from .recovery import (
    held_out_trajectory_mse,
    rate_surface_metrics,
    resolved_expression,
    site_structure_matches,
    strict_structure_match,
)
from .sass import SassAssemblyError, specialize_cubin
from .shape import KernelShape, PackedDispatch
from .template import generate_cuda
from .lm_specialization import specialize_split_lm_cubin
from .lm_selection import (
    SELECTION_NOVELTY_RANDOM,
    SELECTION_OBJECTIVE,
    SELECTION_RANDOM,
    lm_run_diagnostics,
    lm_selection_summary,
    select_lm_candidates,
)
from .trajectory_lm import (
    TrajectoryLMQueue,
    make_lm_population,
    promoted_values,
    render_trajectory_lm_source,
)


def _write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if isinstance(data, str):
        temporary.write_text(data)
    else:
        temporary.write_bytes(data)
    temporary.replace(path)


def _materialized_replay_verdict(search_mse: float, materialized_mse: float) -> tuple[float, float, bool]:
    error = abs(search_mse - materialized_mse)
    tolerance = 1.0e-5 + 1.0e-4 * max(abs(search_mse), abs(materialized_mse))
    passed = (
        math.isfinite(search_mse)
        and math.isfinite(materialized_mse)
        and search_mse < 0.99 * INVALID_MSE
        and materialized_mse < 0.99 * INVALID_MSE
        and error <= tolerance
    )
    return error, tolerance, passed


def _plan_dictionary(plan) -> dict[str, object]:
    return {
        "architecture": f"sm_{plan.architecture}",
        "cubin_bytes": plan.cubin_size,
        "kernel_file_offset": plan.function.file_offset,
        "kernel_bytes": plan.function.size,
        "input_count": plan.input_count,
        "output_count": plan.output_count,
        "patch_capacity": plan.patch_capacity,
        "register_count": plan.register_count,
        "register_count_file_offsets": list(plan.register_count_offsets),
        "register_count_header_file_offsets": list(plan.register_count_header_offsets),
        "load_fence_offset": plan.site.load_fence_offset,
        "site_start": plan.site.start_offset,
        "site_instructions": plan.site.instruction_count,
        "site_end": plan.site.start_offset + 16 * plan.site.instruction_count,
        "incoming_wait_mask": plan.site.incoming_wait_mask,
        "input_registers": list(plan.site.input_registers),
        "output_registers": list(plan.site.output_registers),
        "available_registers": list(plan.site.available_registers),
    }


def _packed_plan_dictionary(plan) -> dict[str, object]:
    return {
        "architecture": f"sm_{plan.architecture}",
        "cubin_bytes": plan.cubin_size,
        "kernel_file_offset": plan.function.file_offset,
        "kernel_bytes": plan.function.size,
        "input_count": plan.input_count,
        "output_count": plan.output_count,
        "genome_capacity": plan.genome_capacity,
        "register_count": plan.register_count,
        "register_count_file_offsets": list(plan.register_count_offsets),
        "register_count_header_file_offsets": list(plan.register_count_header_offsets),
        "dispatch_entry": plan.site.entry_offset,
        "dispatch_instruction_offsets": list(plan.site.dispatch_offsets),
        "arena_start": plan.site.arena_start_offset,
        "arena_instructions": plan.site.arena_instruction_count,
        "arena_end": plan.site.arena_end_offset,
        "continuation": plan.site.continuation_offset,
        "incoming_wait_mask": plan.site.incoming_wait_mask,
        "input_registers": list(plan.site.input_registers),
        "output_registers": list(plan.site.output_registers),
        "available_registers": list(plan.site.available_registers),
        "target_table_file_offsets": list(plan.site.target_table_offsets),
        "original_target_values": list(plan.site.original_target_values),
    }


def _cpu_reference_comparison(
    genomes,
    population,
    reference,
    gpu_mse,
    steps_per_observation: int,
    shape: KernelShape,
    all_settings: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    started = time.perf_counter()
    if all_settings:
        cpu_scores = score_packed_population(
            genomes, population, reference, steps_per_observation, shape
        )
        indices = range(len(cpu_scores))
    else:
        cpu_scores = []
        indices = []
        for genome_index, genome in enumerate(genomes):
            constants = [
                population.settings[
                    (genome_index * shape.constant_count + constant) * population.num_settings
                ]
                for constant in range(shape.constant_count)
            ]
            bindings = [
                population.bindings[
                    (genome_index * shape.input_count + leaf) * population.num_settings
                ]
                for leaf in range(shape.input_count)
            ]
            cpu_scores.append(
                score_setting(
                    genome.programs,
                    constants,
                    bindings,
                    reference,
                    steps_per_observation,
                    shape,
                )
            )
            indices.append(genome_index * population.num_settings)
    cpu_seconds = time.perf_counter() - started

    valid_pairs = 0
    matching_invalid_pairs = 0
    invalid_status_mismatches = 0
    absolute_error_sum = 0.0
    relative_error_sum = 0.0
    maximum_error = 0.0
    maximum_relative_error = 0.0
    worst_valid: dict[str, object] | None = None
    worst_relative: dict[str, object] | None = None
    mismatch_sample: dict[str, object] | None = None
    per_genome: dict[int, dict[str, object]] = {}

    for cpu_index, gpu_index in enumerate(indices):
        genome_index = gpu_index // population.num_settings
        setting = gpu_index % population.num_settings
        gpu_value = float(gpu_mse[gpu_index])
        cpu_value = float(cpu_scores[gpu_index] if all_settings else cpu_scores[cpu_index])
        gpu_invalid = not math.isfinite(gpu_value) or gpu_value >= 0.99 * INVALID_MSE
        cpu_invalid = not math.isfinite(cpu_value) or cpu_value >= 0.99 * INVALID_MSE
        record: dict[str, object] = {
            "genome": genome_index,
            "setting": setting,
            "gpu_mse": gpu_value,
            "cpu_mse": cpu_value,
        }
        if gpu_invalid != cpu_invalid:
            invalid_status_mismatches += 1
            record["absolute_error"] = None
            if mismatch_sample is None:
                mismatch_sample = record
            per_genome.setdefault(genome_index, record)
            continue
        if gpu_invalid:
            matching_invalid_pairs += 1
            record["absolute_error"] = 0.0
            per_genome.setdefault(genome_index, record)
            continue

        error = abs(gpu_value - cpu_value)
        relative_error = error / max(abs(gpu_value), abs(cpu_value), 1.0e-30)
        record["absolute_error"] = error
        record["relative_error"] = relative_error
        valid_pairs += 1
        absolute_error_sum += error
        relative_error_sum += relative_error
        if worst_valid is None or error > maximum_error:
            maximum_error = error
            worst_valid = record
        if worst_relative is None or relative_error > maximum_relative_error:
            maximum_relative_error = relative_error
            worst_relative = record
        previous = per_genome.get(genome_index)
        if previous is None or previous["absolute_error"] is None or error > previous["absolute_error"]:
            per_genome[genome_index] = record

    summary: dict[str, object] = {
        "mode": "all_settings" if all_settings else "setting_zero",
        "checked_configurations": len(indices),
        "cpu_seconds": cpu_seconds,
        "valid_pairs": valid_pairs,
        "matching_invalid_pairs": matching_invalid_pairs,
        "invalid_status_mismatches": invalid_status_mismatches,
        "mean_absolute_error": absolute_error_sum / valid_pairs if valid_pairs else None,
        "max_absolute_error": maximum_error if valid_pairs else None,
        "mean_relative_error": relative_error_sum / valid_pairs if valid_pairs else None,
        "max_relative_error": maximum_relative_error if valid_pairs else None,
        "worst_valid_pair": worst_valid,
        "worst_relative_pair": worst_relative,
        "invalid_mismatch_sample": mismatch_sample,
    }
    return summary, [per_genome[index] for index in sorted(per_genome)]


def _load_model(specification: str) -> SystemModel:
    if specification == "fed_batch":
        return FED_BATCH_MODEL
    if ":" not in specification:
        raise ValueError("model must be 'fed_batch' or a Python module:attribute reference")
    module_name, attribute_name = specification.rsplit(":", 1)
    value = getattr(importlib.import_module(module_name), attribute_name)
    if not isinstance(value, SystemModel):
        raise TypeError(f"{specification} is not a SystemModel")
    return value


def _shape(arguments: argparse.Namespace, model: SystemModel | None = None) -> KernelShape:
    selected = _load_model(arguments.model) if model is None else model
    return selected.make_shape(
        constant_count=arguments.constants,
        trajectory_count=arguments.trajectories,
        observation_count=arguments.observations,
        patch_capacity=arguments.patch_capacity,
    )


def command_generate(arguments: argparse.Namespace) -> None:
    model = _load_model(arguments.model)
    source = generate_cuda(model, _shape(arguments, model))
    if arguments.output == "-":
        sys.stdout.write(source)
    else:
        _write(Path(arguments.output), source)


def command_compile(arguments: argparse.Namespace) -> None:
    model = _load_model(arguments.model)
    shape = _shape(arguments, model)
    source = generate_cuda(model, shape)
    result = compile_cuda(source, arguments.arch, arguments.nvcc)
    output = Path(arguments.output)
    _write(output, result.cubin)
    _write(output.with_suffix(".cu"), source)
    plan = inspect_cubin(result.cubin, shape)
    report = _plan_dictionary(plan)
    report["compile_seconds"] = result.elapsed_seconds
    _write(output.with_suffix(".json"), json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def command_inspect(arguments: argparse.Namespace) -> None:
    plan = inspect_cubin(Path(arguments.cubin).read_bytes(), kernel_name=arguments.kernel)
    rendered = json.dumps(_plan_dictionary(plan), indent=2) + "\n"
    if arguments.output == "-":
        sys.stdout.write(rendered)
    else:
        _write(Path(arguments.output), rendered)


def command_specialize(arguments: argparse.Namespace) -> None:
    cubin = Path(arguments.cubin).read_bytes()
    plan = inspect_cubin(cubin)
    result = specialize_cubin(cubin, plan, planted_programs())
    _write(Path(arguments.output), result.cubin)
    report = _plan_dictionary(plan)
    report.update(
        specialized_instruction_counts=list(result.instruction_counts),
        specialized_register_count=result.register_count,
    )
    print(json.dumps(report, indent=2))


def command_run(arguments: argparse.Namespace) -> None:
    if arguments.model != "fed_batch":
        raise ValueError("fedbatch-run only accepts --model fed_batch")
    model = FED_BATCH_MODEL
    shape = _shape(arguments, model)
    build = Path(arguments.build_directory)
    source = generate_cuda(model, shape)
    _write(build / "fedbatch_template.cu", source)
    compilation = compile_cuda(source, arguments.arch, arguments.nvcc)
    _write(build / "fedbatch_template.cubin", compilation.cubin)
    inspected_started = time.perf_counter()
    plan = inspect_cubin(compilation.cubin, shape)
    inspection_seconds = time.perf_counter() - inspected_started
    specialized_started = time.perf_counter()
    specialization = specialize_cubin(compilation.cubin, plan, planted_programs())
    specialization_seconds = time.perf_counter() - specialized_started
    _write(build / "fedbatch_specialized.cubin", specialization.cubin)

    population = make_population(arguments.settings, shape, arguments.seed)
    reference = reference_data(arguments.steps)
    result = run_fedbatch(
        specialization.cubin,
        population,
        reference,
        shape,
        arguments.steps,
        arguments.threads,
        arguments.repetitions,
        arguments.device,
    )
    best_setting = min(range(population.num_settings), key=result.mse.__getitem__)
    planted_rank = sorted(range(population.num_settings), key=result.mse.__getitem__).index(0) + 1
    report = {
        **_plan_dictionary(plan),
        "compile_seconds": compilation.elapsed_seconds,
        "inspection_seconds": inspection_seconds,
        "specialization_seconds": specialization_seconds,
        "specialized_instruction_counts": list(specialization.instruction_counts),
        "specialized_register_count": specialization.register_count,
        "settings": population.num_settings,
        "steps_per_observation": arguments.steps,
        "threads_per_block": arguments.threads,
        "seconds_per_launch": result.seconds_per_launch,
        "module_load_seconds": result.module_load_seconds,
        "function_lookup_seconds": result.function_lookup_seconds,
        "settings_per_second": population.num_settings / result.seconds_per_launch,
        "trajectories_per_second": 3 * population.num_settings / result.seconds_per_launch,
        "planted_mse": result.mse[0],
        "planted_rank": planted_rank,
        "best_setting": best_setting,
        "best_mse": result.mse[best_setting],
    }
    _write(build / "run.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def command_packed_run(arguments: argparse.Namespace) -> None:
    if arguments.model != "fed_batch":
        raise ValueError("fedbatch-packed-run only accepts --model fed_batch")
    shape = _shape(arguments, FED_BATCH_MODEL)
    dispatch = PackedDispatch(arguments.genome_capacity, arguments.genomes_per_cta)
    if arguments.genomes > dispatch.genome_capacity:
        raise ValueError("--genomes cannot exceed --genome-capacity")
    genomes = benchmark_genomes(arguments.genomes, shape)
    build = Path(arguments.build_directory)
    source = generate_packed_cuda(FED_BATCH_MODEL, shape, dispatch)
    _write(build / "fedbatch_packed_template.cu", source)
    compilation = compile_cuda(source, arguments.arch, arguments.nvcc)
    _write(build / "fedbatch_packed_template.cubin", compilation.cubin)
    inspected_started = time.perf_counter()
    plan = inspect_packed_cubin(compilation.cubin, shape, dispatch)
    inspection_seconds = time.perf_counter() - inspected_started
    specialized_started = time.perf_counter()
    specialization = specialize_packed_cubin(
        compilation.cubin, plan, genomes, shape
    )
    specialization_seconds = time.perf_counter() - specialized_started
    _write(build / "fedbatch_packed_specialized.cubin", specialization.cubin)

    base_population = make_population(arguments.settings, shape, arguments.seed)
    population = pack_population(base_population, len(genomes), shape)
    reference = reference_data(arguments.steps)
    result = run_packed_fedbatch(
        specialization.cubin,
        population,
        reference,
        shape,
        dispatch,
        arguments.steps,
        arguments.threads,
        arguments.repetitions,
        arguments.device,
    )
    cpu_reference, comparisons = _cpu_reference_comparison(
        genomes,
        population,
        reference,
        result.mse,
        arguments.steps,
        shape,
        arguments.cpu_check_all_settings,
    )

    setting_tiles = (population.num_settings + arguments.threads - 1) // arguments.threads
    genome_groups = (
        population.num_genomes + dispatch.genomes_per_cta - 1
    ) // dispatch.genomes_per_cta
    configurations = population.num_genomes * population.num_settings
    report = {
        **_packed_plan_dictionary(plan),
        "compile_seconds": compilation.elapsed_seconds,
        "inspection_seconds": inspection_seconds,
        "specialization_seconds": specialization_seconds,
        "specialized_instruction_counts": [
            list(counts) for counts in specialization.instruction_counts
        ],
        "body_instruction_counts": list(specialization.body_instruction_counts),
        "body_offsets": list(specialization.body_offsets),
        "specialized_register_count": specialization.register_count,
        "genomes": population.num_genomes,
        "genomes_per_cta": dispatch.genomes_per_cta,
        "settings": population.num_settings,
        "steps_per_observation": arguments.steps,
        "threads_per_block": arguments.threads,
        "grid": [setting_tiles, genome_groups, 1],
        "cta_count": setting_tiles * genome_groups,
        "seconds_per_launch": result.seconds_per_launch,
        "module_load_seconds": result.module_load_seconds,
        "function_lookup_seconds": result.function_lookup_seconds,
        "configurations_per_second": configurations / result.seconds_per_launch,
        "trajectories_per_second": (
            shape.trajectory_count * configurations / result.seconds_per_launch
        ),
        "genome_zero_planted_mse": result.mse[0],
        "cpu_reference": cpu_reference,
        "correctness": comparisons,
    }
    _write(build / "packed_run.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def command_packed_module_generate(arguments: argparse.Namespace) -> None:
    model = _load_model(arguments.model)
    shape = _shape(arguments, model)
    dispatch = PackedDispatch(arguments.genome_capacity, arguments.genomes_per_cta)
    source = generate_packed_cuda_module(model, shape, arguments.kernels, dispatch)
    if arguments.output == "-":
        sys.stdout.write(source)
    else:
        _write(Path(arguments.output), source)


def command_packed_module_run(arguments: argparse.Namespace) -> None:
    if arguments.model != "fed_batch":
        raise ValueError("fedbatch-module-run only accepts --model fed_batch")
    shape = _shape(arguments, FED_BATCH_MODEL)
    dispatch = PackedDispatch(arguments.genome_capacity, arguments.genomes_per_cta)
    module_capacity = arguments.kernels * dispatch.genome_capacity
    if arguments.genomes > module_capacity:
        raise ValueError("--genomes exceeds the combined module genome capacity")

    genomes = benchmark_genomes(arguments.genomes, shape)
    build = Path(arguments.build_directory)
    source = generate_packed_cuda_module(
        FED_BATCH_MODEL, shape, arguments.kernels, dispatch
    )
    _write(build / "fedbatch_module_template.cu", source)
    compilation = compile_cuda(source, arguments.arch, arguments.nvcc)
    _write(build / "fedbatch_module_template.cubin", compilation.cubin)

    inspected_started = time.perf_counter()
    inspection = inspect_module(source, compilation.cubin)
    inspection_seconds = time.perf_counter() - inspected_started
    if not isinstance(inspection.plan, PackedModulePlan):
        raise RuntimeError("module inspection did not return a packed module plan")
    _write(build / "fedbatch_module_inspection.json", json.dumps(inspection.document, indent=2) + "\n")

    specialized_started = time.perf_counter()
    specialization = specialize_packed_module_cubin(
        compilation.cubin, inspection.plan, genomes, shape
    )
    specialization_seconds = time.perf_counter() - specialized_started
    _write(build / "fedbatch_module_specialized.cubin", specialization.cubin)

    base_population = make_population(arguments.settings, shape, arguments.seed)
    population = pack_population(base_population, len(genomes), shape)
    reference = reference_data(arguments.steps)
    result = run_packed_module_fedbatch(
        specialization.cubin,
        inspection.plan,
        population,
        reference,
        shape,
        arguments.steps,
        arguments.threads,
        arguments.repetitions,
        arguments.device,
    )

    cpu_reference, comparisons = _cpu_reference_comparison(
        genomes,
        population,
        reference,
        result.mse,
        arguments.steps,
        shape,
        arguments.cpu_check_all_settings,
    )

    configurations = population.num_genomes * population.num_settings
    report = {
        "architecture": f"sm_{inspection.plan.architecture}",
        "compile_seconds": compilation.elapsed_seconds,
        "inspection_seconds": inspection_seconds,
        "specialization_seconds": specialization_seconds,
        "module_load_seconds": result.module_load_seconds,
        "function_lookup_seconds": result.function_lookup_seconds,
        "module_bytes": len(specialization.cubin),
        "kernel_count": len(inspection.plan.kernels),
        "active_kernel_count": result.kernel_count,
        "module_genome_capacity": inspection.plan.genome_capacity,
        "genomes": population.num_genomes,
        "genome_capacity_per_kernel": dispatch.genome_capacity,
        "genomes_per_cta": dispatch.genomes_per_cta,
        "settings": population.num_settings,
        "steps_per_observation": arguments.steps,
        "threads_per_block": arguments.threads,
        "seconds_per_module_launch": result.seconds_per_launch,
        "configurations_per_second": configurations / result.seconds_per_launch,
        "trajectories_per_second": shape.trajectory_count * configurations / result.seconds_per_launch,
        "cpu_reference": cpu_reference,
        "kernels": [
            {
                "name": kernel.name,
                "genome_base": kernel.genome_base,
                "genome_count": kernel.genome_count,
                "specialized_register_count": kernel.register_count,
                "body_instruction_counts": list(kernel.body_instruction_counts),
                "body_offsets": list(kernel.body_offsets),
                "specialized_instruction_counts": [
                    list(counts) for counts in kernel.instruction_counts
                ],
            }
            for kernel in specialization.kernels
        ],
        "correctness": comparisons,
    }
    _write(build / "module_run.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _worker_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("worker counts must be comma-separated integers") from exc
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("worker counts must be positive")
    return counts


def _positive_float_counts(value: str) -> tuple[float, ...]:
    try:
        counts = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not counts or any(not math.isfinite(count) or count <= 0.0 for count in counts):
        raise argparse.ArgumentTypeError("thresholds must be finite and positive")
    return counts


def command_c99_pipeline_run(arguments: argparse.Namespace) -> None:
    if arguments.model != "fed_batch":
        raise ValueError("fedbatch-c99-pipeline-run only accepts --model fed_batch")
    shape = _shape(arguments, FED_BATCH_MODEL)
    if shape.trajectory_count != 3 or shape.observation_count != 12:
        raise ValueError("the current fed-batch reference requires 3 trajectories and 12 observations")
    dispatch = PackedDispatch(arguments.genome_capacity, arguments.genomes_per_cta)
    module_capacity = arguments.kernels * dispatch.genome_capacity
    if arguments.genomes > module_capacity:
        raise ValueError("--genomes exceeds the combined module genome capacity")
    genomes = benchmark_genomes(arguments.genomes, shape)
    batch = FlatGenomeBatch.from_genomes(genomes, shape)
    build = Path(arguments.build_directory)
    source = generate_packed_cuda_module(FED_BATCH_MODEL, shape, arguments.kernels, dispatch)
    _write(build / "fedbatch_c99_template.cu", source)
    compilation = compile_cuda(source, arguments.arch, arguments.nvcc)
    _write(build / "fedbatch_c99_template.cubin", compilation.cubin)
    inspection = inspect_module(source, compilation.cubin)
    if not isinstance(inspection.plan, PackedModulePlan):
        raise RuntimeError("C99 pipeline benchmark requires a packed module plan")
    api = C99Library(arguments.c99_library)
    reference = reference_data(arguments.steps)
    base_population = make_population(arguments.settings, shape, arguments.seed)
    population = pack_population(base_population, arguments.genomes, shape)
    with api.template(compilation.cubin, inspection.plan, shape) as template:
        python_started = time.perf_counter()
        python_specialization = specialize_packed_module_cubin(compilation.cubin, inspection.plan, genomes, shape)
        python_specialization_seconds = time.perf_counter() - python_started
        c_started = time.perf_counter()
        c_specialized, c_stats = template.specialize(batch)
        c_specialization_seconds = time.perf_counter() - c_started
        specialization_sweeps = [
            benchmark_specialization(
                template, batch, workers, arguments.specialization_submissions
            )
            for workers in arguments.worker_counts
        ]
        with template.pipeline(
            arguments.workers,
            max(arguments.queue_capacity, arguments.submissions),
            arguments.loaded_modules,
            arguments.device,
            enable_cuda=True,
        ) as pipeline:
            allocations: list[int] = []

            def allocate(values=None, byte_count: int | None = None) -> int:
                size = byte_count if byte_count is not None else len(memoryview(values).cast("B"))
                pointer = pipeline.allocate(size)
                allocations.append(pointer)
                if values is not None:
                    pipeline.upload(pointer, values)
                return pointer

            try:
                settings_device = allocate(population.settings)
                bindings_device = allocate(population.bindings)
                reference_device = allocate(reference)
                output = array("f", [0.0]) * (population.num_genomes * population.num_settings)
                mse_device = allocate(byte_count=len(output) * output.itemsize)
                launch = CFedbatchLaunch(
                    settings_device,
                    population.num_settings,
                    bindings_device,
                    population.num_settings,
                    population.num_settings,
                    population.num_genomes,
                    reference_device,
                    arguments.steps,
                    mse_device,
                    arguments.threads,
                    shape.shared_bytes(arguments.threads),
                )
                warmup = pipeline.submit(batch, launch)
                warmup.wait()
                warmup.close()
                started = time.perf_counter()
                tickets = [pipeline.submit(batch, launch) for _ in range(arguments.submissions)]
                ticket_results = [ticket.wait() for ticket in tickets]
                pipeline_seconds = time.perf_counter() - started
                for ticket in tickets:
                    ticket.close()
                pipeline.download(mse_device, output)
            finally:
                for allocation in reversed(allocations):
                    pipeline.free(allocation)
    cpu_reference, comparisons = _cpu_reference_comparison(
        genomes,
        population,
        reference,
        output,
        arguments.steps,
        shape,
        arguments.cpu_check_all_settings,
    )
    configurations = arguments.submissions * population.num_genomes * population.num_settings
    report = {
        "architecture": f"sm_{inspection.plan.architecture}",
        "compile_seconds": compilation.elapsed_seconds,
        "module_bytes": len(compilation.cubin),
        "kernel_count": len(inspection.plan.kernels),
        "genomes": batch.genome_count,
        "asts": batch.ast_count,
        "postorder_bytes": batch.program_byte_count,
        "settings": population.num_settings,
        "steps_per_observation": arguments.steps,
        "threads_per_block": arguments.threads,
        "python_specialization_seconds": python_specialization_seconds,
        "c99_specialization_seconds": c_specialization_seconds,
        "c99_matches_python": c_specialized == python_specialization.cubin,
        "specialized_sass_instructions": c_stats.sass_instruction_count,
        "specialized_maximum_register_count": c_stats.maximum_register_count,
        "specialization_worker_sweep": [result.__dict__ for result in specialization_sweeps],
        "eager_pipeline": {
            "workers": arguments.workers,
            "queue_capacity": max(arguments.queue_capacity, arguments.submissions),
            "maximum_loaded_modules": arguments.loaded_modules,
            "submissions": arguments.submissions,
            "wall_seconds": pipeline_seconds,
            "modules_per_second": arguments.submissions / pipeline_seconds,
            "configurations_per_second": configurations / pipeline_seconds,
            "trajectories_per_second": shape.trajectory_count * configurations / pipeline_seconds,
            "mean_specialization_seconds": sum(item.specialization_seconds for item in ticket_results) / len(ticket_results),
            "mean_module_load_seconds": sum(item.module_load_seconds for item in ticket_results) / len(ticket_results),
            "mean_function_lookup_seconds": sum(item.function_lookup_seconds for item in ticket_results) / len(ticket_results),
            "mean_launch_seconds": sum(item.launch_seconds for item in ticket_results) / len(ticket_results),
        },
        "cpu_reference": cpu_reference,
        "correctness": comparisons,
    }
    _write(build / "c99_pipeline_run.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _winner_correctness(
    genomes,
    population,
    reference,
    full_gpu_scores,
    winner_scores,
    winner_settings,
    submission_index: int,
    steps_per_observation: int,
    shape: KernelShape,
    cpu_score_cache: dict[tuple[int, int], float] | None = None,
) -> dict[str, object]:
    exact_setting_matches = 0
    maximum_gpu_score_error = 0.0
    maximum_cpu_score_error = 0.0
    invalid_status_mismatches = 0
    samples = []
    output_base = submission_index * population.num_genomes
    for genome_index, genome in enumerate(genomes):
        score_base = genome_index * population.num_settings
        expected_setting = min(
            range(population.num_settings),
            key=lambda setting: (float(full_gpu_scores[score_base + setting]), setting),
        )
        observed_setting = int(winner_settings[output_base + genome_index])
        observed_score = float(winner_scores[output_base + genome_index])
        expected_score = float(full_gpu_scores[score_base + expected_setting])
        if observed_setting == expected_setting:
            exact_setting_matches += 1
        maximum_gpu_score_error = max(
            maximum_gpu_score_error, abs(observed_score - expected_score)
        )
        if observed_setting >= population.num_settings:
            invalid_status_mismatches += 1
            cpu_score = INVALID_MSE
        else:
            cache_key = (genome_index, observed_setting)
            if cpu_score_cache is not None and cache_key in cpu_score_cache:
                cpu_score = cpu_score_cache[cache_key]
            else:
                constants, bindings = packed_setting_values(
                    population, genome_index, observed_setting, shape
                )
                cpu_score = score_setting(
                    genome.programs,
                    constants,
                    bindings,
                    reference,
                    steps_per_observation,
                    shape,
                )
                if cpu_score_cache is not None:
                    cpu_score_cache[cache_key] = cpu_score
            gpu_invalid = not math.isfinite(observed_score) or observed_score >= 0.99 * INVALID_MSE
            cpu_invalid = not math.isfinite(cpu_score) or cpu_score >= 0.99 * INVALID_MSE
            if gpu_invalid != cpu_invalid:
                invalid_status_mismatches += 1
            elif not gpu_invalid:
                maximum_cpu_score_error = max(
                    maximum_cpu_score_error, abs(observed_score - cpu_score)
                )
        if genome_index < 8:
            samples.append(
                {
                    "genome": genome_index,
                    "expected_setting": expected_setting,
                    "winner_setting": observed_setting,
                    "full_gpu_score": expected_score,
                    "winner_gpu_score": observed_score,
                    "cpu_score": cpu_score,
                }
            )
    return {
        "checked_genomes": len(genomes),
        "exact_setting_matches": exact_setting_matches,
        "maximum_gpu_score_error": maximum_gpu_score_error,
        "maximum_cpu_score_error": maximum_cpu_score_error,
        "invalid_status_mismatches": invalid_status_mismatches,
        "samples": samples,
    }


def command_c99_unique_pipeline_sweep(arguments: argparse.Namespace) -> None:
    if arguments.model != "fed_batch":
        raise ValueError("fedbatch-c99-unique-sweep only accepts --model fed_batch")
    if arguments.workers != 1:
        raise ValueError("the ordered unique-module benchmark currently requires one worker")
    shape = _shape(arguments, FED_BATCH_MODEL)
    if shape.trajectory_count != 3 or shape.observation_count != 12:
        raise ValueError("the current fed-batch reference requires 3 trajectories and 12 observations")
    dispatch = PackedDispatch(arguments.genome_capacity, arguments.genomes_per_cta)
    if arguments.warp_per_system:
        if any(setting_count > 32 for setting_count in arguments.settings_counts):
            raise ValueError("--warp-per-system currently supports at most 32 settings")
        if arguments.threads % 32 != 0:
            raise ValueError("--warp-per-system requires a whole number of warps per CTA")
        if arguments.genomes_per_cta > arguments.threads // 32:
            raise ValueError("--genomes-per-cta exceeds the available warps")
    module_capacity = arguments.kernels * dispatch.genome_capacity
    if arguments.genomes > module_capacity:
        raise ValueError("--genomes exceeds the combined module genome capacity")

    build = Path(arguments.build_directory)
    source = generate_packed_cuda_module(
        FED_BATCH_MODEL,
        shape,
        arguments.kernels,
        dispatch,
        winner_output=arguments.winner_output,
        warp_per_system=arguments.warp_per_system,
    )
    _write(build / "fedbatch_unique_template.cu", source)
    compilation = compile_cuda(source, arguments.arch, arguments.nvcc)
    _write(build / "fedbatch_unique_template.cubin", compilation.cubin)
    inspection = inspect_module(source, compilation.cubin)
    if not isinstance(inspection.plan, PackedModulePlan):
        raise RuntimeError("unique C99 benchmark requires a packed module plan")
    full_reference = None
    if arguments.winner_output:
        full_source = generate_packed_cuda_module(
            FED_BATCH_MODEL,
            shape,
            arguments.kernels,
            dispatch,
        )
        full_compilation = compile_cuda(full_source, arguments.arch, arguments.nvcc)
        full_inspection = inspect_module(full_source, full_compilation.cubin)
        if not isinstance(full_inspection.plan, PackedModulePlan):
            raise RuntimeError("full-score reference requires a packed module plan")
        full_reference = (full_compilation, full_inspection.plan)

    batch_generation_started = time.perf_counter()
    genome_fingerprints: set[tuple[bytes, ...]] = set()
    ast_fingerprints: tuple[set[bytes], ...] = tuple(
        set() for _ in range(shape.ast_count)
    )
    workload_hasher = hashlib.sha256()
    duplicate_genome_retries = 0

    def make_benchmark_genomes(
        variant: int,
        require_unique: bool = False,
    ) -> tuple[SystemGenome, ...]:
        nonlocal duplicate_genome_retries
        if not arguments.structurally_diverse:
            genomes = benchmark_genomes(arguments.genomes, shape, variant=variant)
            if require_unique:
                for genome in genomes:
                    fingerprint = tuple(program.data for program in genome.programs)
                    if fingerprint in genome_fingerprints or any(
                        program.data in ast_fingerprints[site]
                        for site, program in enumerate(genome.programs)
                    ):
                        raise RuntimeError(
                            "the benchmark generated a duplicate system genome or AST"
                        )
                    genome_fingerprints.add(fingerprint)
                    for site, program in enumerate(genome.programs):
                        ast_fingerprints[site].add(program.data)
            return genomes
        genomes: list[SystemGenome] = []
        for genome_index in range(arguments.genomes):
            sequence_index = (
                (variant - 1) * arguments.genomes + genome_index
                if require_unique
                else (1 << 63) + genome_index
            )
            random_source = random.Random(
                (int(arguments.seed) << 32)
                ^ (sequence_index * 0x9E3779B97F4A7C15)
                ^ 0x53534944554E4951
            )
            case = sequence_index
            for attempt in range(4096):
                genome = random_genome(random_source, shape, case + attempt)
                fingerprint = tuple(program.data for program in genome.programs)
                unique_asts = all(
                    program.data not in ast_fingerprints[site]
                    for site, program in enumerate(genome.programs)
                )
                if not require_unique or (
                    fingerprint not in genome_fingerprints and unique_asts
                ):
                    if require_unique:
                        genome_fingerprints.add(fingerprint)
                        for site, program in enumerate(genome.programs):
                            ast_fingerprints[site].add(program.data)
                        duplicate_genome_retries += attempt
                        for site, program in enumerate(genome.programs):
                            workload_hasher.update(site.to_bytes(2, "little"))
                            workload_hasher.update(len(program.data).to_bytes(4, "little"))
                            workload_hasher.update(program.data)
                    genomes.append(genome)
                    break
            else:
                raise RuntimeError(
                    "could not generate a distinct structurally diverse system genome"
                )
        return tuple(genomes)

    warmup_genomes = make_benchmark_genomes(0)
    warmup_batch = FlatGenomeBatch.from_genomes(warmup_genomes, shape)
    batches: list[FlatGenomeBatch] = []
    final_genomes = None
    for variant in range(1, arguments.submissions + 1):
        genomes = make_benchmark_genomes(variant, require_unique=True)
        batches.append(FlatGenomeBatch.from_genomes(genomes, shape))
        if variant == arguments.submissions:
            final_genomes = genomes
    batch_generation_seconds = time.perf_counter() - batch_generation_started
    if final_genomes is None:
        raise RuntimeError("the unique batch set is empty")

    api = C99Library(arguments.c99_library)
    reference = reference_data(arguments.steps)
    sweep: list[dict[str, object]] = []
    correctness_by_settings: dict[str, object] = {}
    with api.template(compilation.cubin, inspection.plan, shape) as template:
        hash_started = time.perf_counter()
        hashes = []
        register_counts = set()
        instruction_counts = set()
        for batch in batches:
            specialized, stats = template.specialize(batch)
            hashes.append(hashlib.sha256(specialized).hexdigest())
            register_counts.add(int(stats.maximum_register_count))
            instruction_counts.add(int(stats.sass_instruction_count))
        hash_verification_seconds = time.perf_counter() - hash_started
        unique_hashes = len(set(hashes))
        if unique_hashes != len(batches):
            raise RuntimeError(
                f"only {unique_hashes} of {len(batches)} specialized CUBIN hashes are unique"
            )

        for setting_count in arguments.settings_counts:
            base_population = make_population(setting_count, shape, arguments.seed)
            population = pack_population(base_population, arguments.genomes, shape)
            outputs_by_configuration: list[tuple[int, int, object]] = []
            for execution_streams in arguments.execution_stream_counts:
                for loaded_modules in arguments.loaded_module_depths:
                    if loaded_modules < execution_streams:
                        raise ValueError(
                            "--loaded-module-depths values must be at least every execution stream count"
                        )
                    with template.pipeline(
                        arguments.workers,
                        arguments.submissions,
                        loaded_modules,
                        arguments.device,
                        enable_cuda=True,
                        execution_streams=execution_streams,
                    ) as pipeline:
                        allocations: list[int] = []

                        def allocate(values=None, byte_count: int | None = None) -> int:
                            size = byte_count if byte_count is not None else len(memoryview(values).cast("B"))
                            pointer = pipeline.allocate(size)
                            allocations.append(pointer)
                            if values is not None:
                                pipeline.upload(pointer, values)
                            return pointer

                        try:
                            settings_device = allocate(population.settings)
                            bindings_device = allocate(population.bindings)
                            reference_device = allocate(reference)
                            if arguments.winner_output:
                                settings_per_tile = (
                                    32 if arguments.warp_per_system else arguments.threads
                                )
                                setting_tiles = (
                                    population.num_settings + settings_per_tile - 1
                                ) // settings_per_tile
                                cta_stride = population.num_genomes * setting_tiles * 4
                                winner_stride = population.num_genomes * 4
                                cta_score_device = allocate(
                                    byte_count=arguments.submissions * cta_stride
                                )
                                cta_setting_device = allocate(
                                    byte_count=arguments.submissions * cta_stride
                                )
                                winner_score_device = allocate(
                                    byte_count=arguments.submissions * winner_stride
                                )
                                winner_setting_device = allocate(
                                    byte_count=arguments.submissions * winner_stride
                                )
                                winner_scores = array("f", [0.0]) * (
                                    arguments.submissions * population.num_genomes
                                )
                                winner_settings = array("I", [0]) * (
                                    arguments.submissions * population.num_genomes
                                )

                                def make_launch(index: int) -> CFedbatchLaunch:
                                    return CFedbatchLaunch(
                                        settings_device,
                                        population.num_settings,
                                        bindings_device,
                                        population.num_settings,
                                        population.num_settings,
                                        population.num_genomes,
                                        reference_device,
                                        arguments.steps,
                                        0,
                                        arguments.threads,
                                        (
                                            shape.shared_bytes(arguments.threads)
                                            if arguments.warp_per_system
                                            else shape.winner_shared_bytes(arguments.threads)
                                        ),
                                        SSID_OUTPUT_GENOME_WINNERS,
                                        arguments.reduction_threads,
                                        cta_score_device + index * cta_stride,
                                        cta_setting_device + index * cta_stride,
                                        winner_score_device + index * winner_stride,
                                        winner_setting_device + index * winner_stride,
                                    )

                                launch = make_launch(0)
                                launches = [
                                    make_launch(index)
                                    for index in range(arguments.submissions)
                                ]
                            else:
                                output = array("f", [0.0]) * (
                                    population.num_genomes * population.num_settings
                                )
                                mse_device = allocate(byte_count=len(output) * output.itemsize)
                                launch = CFedbatchLaunch(
                                    settings_device,
                                    population.num_settings,
                                    bindings_device,
                                    population.num_settings,
                                    population.num_settings,
                                    population.num_genomes,
                                    reference_device,
                                    arguments.steps,
                                    mse_device,
                                    arguments.threads,
                                    shape.shared_bytes(arguments.threads),
                                )
                                launches = [launch] * arguments.submissions
                            warmup = pipeline.submit(warmup_batch, launch)
                            warmup.wait()
                            warmup.close()
                            started = time.perf_counter()
                            tickets = [
                                pipeline.submit(batch, batch_launch)
                                for batch, batch_launch in zip(batches, launches)
                            ]
                            ticket_results = [ticket.wait() for ticket in tickets]
                            for ticket in tickets:
                                ticket.close()
                            if arguments.winner_output:
                                pipeline.download(winner_score_device, winner_scores)
                                pipeline.download(winner_setting_device, winner_settings)
                                final_output = (winner_scores, winner_settings)
                            else:
                                pipeline.download(mse_device, output)
                                final_output = output
                            wall_seconds = time.perf_counter() - started
                        finally:
                            for allocation in reversed(allocations):
                                pipeline.free(allocation)

                    outputs_by_configuration.append(
                        (execution_streams, loaded_modules, final_output)
                    )
                    configurations = (
                        arguments.submissions * population.num_genomes * population.num_settings
                    )
                    load_times = [result.module_load_seconds for result in ticket_results]
                    total_times = [result.total_seconds for result in ticket_results]
                    specialization_times = [
                        result.specialization_seconds for result in ticket_results
                    ]
                    sweep.append(
                        {
                            "settings": setting_count,
                            "execution_streams": execution_streams,
                            "maximum_loaded_modules": loaded_modules,
                            "submissions": arguments.submissions,
                            "wall_seconds": wall_seconds,
                            "modules_per_second": arguments.submissions / wall_seconds,
                            "unique_genomes_per_second": (
                                arguments.submissions * population.num_genomes / wall_seconds
                            ),
                            "unique_candidate_asts_per_second": (
                                arguments.submissions * population.num_genomes * shape.ast_count
                                / wall_seconds
                            ),
                            "configurations_per_second": configurations / wall_seconds,
                            "trajectories_per_second": (
                                shape.trajectory_count * configurations / wall_seconds
                            ),
                            "mean_specialization_seconds": sum(specialization_times)
                            / len(specialization_times),
                            "mean_module_load_seconds": sum(load_times) / len(load_times),
                            "p50_module_load_seconds": _percentile(load_times, 0.50),
                            "p95_module_load_seconds": _percentile(load_times, 0.95),
                            "p99_module_load_seconds": _percentile(load_times, 0.99),
                            "p50_ticket_seconds": _percentile(total_times, 0.50),
                            "p95_ticket_seconds": _percentile(total_times, 0.95),
                            "p99_ticket_seconds": _percentile(total_times, 0.99),
                        }
                    )

            if arguments.winner_output:
                full_compilation, full_plan = full_reference
                full_specialization = specialize_packed_module_cubin(
                    full_compilation.cubin, full_plan, final_genomes, shape
                )
                full_run = run_packed_module_fedbatch(
                    full_specialization.cubin,
                    full_plan,
                    population,
                    reference,
                    shape,
                    arguments.steps,
                    arguments.threads,
                    1,
                    arguments.device,
                )
                cpu_score_cache: dict[tuple[int, int], float] = {}
                for execution_streams, loaded_modules, final_output in outputs_by_configuration:
                    winner_scores, winner_settings = final_output
                    key = (
                        f"settings={setting_count},streams={execution_streams},"
                        f"loaded={loaded_modules}"
                    )
                    correctness_by_settings[key] = _winner_correctness(
                        final_genomes,
                        population,
                        reference,
                        full_run.mse,
                        winner_scores,
                        winner_settings,
                        arguments.submissions - 1,
                        arguments.steps,
                        shape,
                        cpu_score_cache,
                    )
            else:
                for execution_streams, loaded_modules, final_output in outputs_by_configuration:
                    cpu_reference, comparisons = _cpu_reference_comparison(
                        final_genomes,
                        population,
                        reference,
                        final_output,
                        arguments.steps,
                        shape,
                        False,
                    )
                    key = (
                        f"settings={setting_count},streams={execution_streams},"
                        f"loaded={loaded_modules}"
                    )
                    correctness_by_settings[key] = {
                        "cpu_reference": cpu_reference,
                        "samples": comparisons,
                    }

    report = {
        "schema": "secant.system_id.c99_unique_pipeline_sweep.v3",
        "output_mode": "genome_winners" if arguments.winner_output else "full_mse",
        "architecture": f"sm_{inspection.plan.architecture}",
        "compile_seconds": compilation.elapsed_seconds,
        "module_bytes": len(compilation.cubin),
        "kernel_count": len(inspection.plan.kernels),
        "genomes_per_module": arguments.genomes,
        "candidate_asts_per_module": arguments.genomes * shape.ast_count,
        "work_ownership": (
            "warp_per_system" if arguments.warp_per_system else "cta_serial"
        ),
        "correctness_reference_ownership": "cta_serial",
        "settings_per_scoring_group": (
            32 if arguments.warp_per_system else arguments.threads
        ),
        "genomes_per_cta": arguments.genomes_per_cta,
        "threads_per_block": arguments.threads,
        "execution_stream_counts": list(arguments.execution_stream_counts),
        "reduction_threads": arguments.reduction_threads if arguments.winner_output else None,
        "steps_per_observation": arguments.steps,
        "batch_generation_seconds": batch_generation_seconds,
        "hash_verification_seconds": hash_verification_seconds,
        "submitted_modules": len(batches),
        "genome_family": (
            "random_smooth_structures"
            if arguments.structurally_diverse
            else "scaled_fixed_rational_structure"
        ),
        "distinct_system_genomes": len(genome_fingerprints),
        "distinct_candidate_asts": sum(len(values) for values in ast_fingerprints),
        "distinct_candidate_asts_by_site": [
            len(values) for values in ast_fingerprints
        ],
        "workload_sha256": workload_hasher.hexdigest(),
        "duplicate_genome_retries": duplicate_genome_retries,
        "unique_cubin_hashes": unique_hashes,
        "first_cubin_sha256": hashes[0],
        "last_cubin_sha256": hashes[-1],
        "specialized_register_counts": sorted(register_counts),
        "specialized_instruction_counts": sorted(instruction_counts),
        "sweep": sweep,
        "correctness": correctness_by_settings,
    }
    _write(build / "unique_pipeline_sweep.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def command_c99_gp_run(arguments: argparse.Namespace) -> None:
    if arguments.model != "fed_batch":
        raise ValueError("fedbatch-c99-gp-run only accepts --model fed_batch")
    shape = _shape(arguments, FED_BATCH_MODEL)
    if shape.trajectory_count != 3 or shape.observation_count != 12:
        raise ValueError("the current fed-batch reference requires 3 trajectories and 12 observations")
    dispatch = PackedDispatch(arguments.genome_capacity, arguments.genomes_per_cta)
    module_capacity = arguments.kernels * dispatch.genome_capacity
    if arguments.population < module_capacity:
        raise ValueError("--population must be at least one complete module")
    build = Path(arguments.build_directory)
    source = generate_packed_cuda_module(
        FED_BATCH_MODEL,
        shape,
        arguments.kernels,
        dispatch,
        winner_output=True,
        hashed_settings=True,
        constant_mutation_scale=arguments.constant_scale,
        binding_keep_probability=arguments.binding_keep,
    )
    _write(build / "fedbatch_gp_template.cu", source)
    compilation = compile_cuda(source, arguments.arch, arguments.nvcc)
    _write(build / "fedbatch_gp_template.cubin", compilation.cubin)
    inspection = inspect_module(source, compilation.cubin)
    if not isinstance(inspection.plan, PackedModulePlan):
        raise RuntimeError("C99 GP benchmark requires a packed module plan")
    config = GPConfig(
        population_size=arguments.population,
        settings_per_genome=arguments.settings,
        max_nodes_per_ast=arguments.max_nodes,
        max_program_bytes_per_ast=arguments.max_program_bytes,
        initial_max_nodes=arguments.initial_max_nodes,
        max_depth=arguments.max_depth,
        tournament_size=arguments.tournament,
        elite_count=arguments.elites,
        seed=arguments.seed,
        subtree_crossover_probability=arguments.subtree_crossover,
        whole_site_crossover_probability=arguments.whole_site_crossover,
        subtree_mutation_probability=arguments.subtree_mutation,
        point_mutation_probability=arguments.point_mutation,
        binding_keep_probability=arguments.binding_keep,
        constant_mutation_scale=arguments.constant_scale,
        parsimony_coefficient=arguments.parsimony,
    )
    unary_operations = () if arguments.no_unary else (
        InstructionType.NEG_F32,
        InstructionType.ABS_F32,
    )
    binary_operations = (
        InstructionType.ADD_F32,
        InstructionType.SUB_F32,
        InstructionType.MUL_F32,
        InstructionType.DIV_F32,
    )
    api = C99Library(arguments.c99_library)
    reference = reference_data(arguments.steps)
    with api.template(
        compilation.cubin,
        inspection.plan,
        shape,
        settings_mode=SSID_SETTINGS_HASHED_INCUMBENT,
    ) as template:
        with template.gp(config, shape, unary_operations, binary_operations) as gp:
            batch_count = gp.batch_count
            with template.pipeline(
                arguments.workers,
                max(batch_count, arguments.loaded_modules),
                arguments.loaded_modules,
                arguments.device,
                enable_cuda=True,
            ) as pipeline:
                reference_device = pipeline.allocate(len(reference) * reference.itemsize)
                try:
                    pipeline.upload(reference_device, reference)
                    stats = gp.run(
                        pipeline,
                        arguments.generations,
                        reference_device,
                        arguments.steps,
                        arguments.threads,
                        arguments.reduction_threads,
                    )
                finally:
                    pipeline.free(reference_device)
            best = gp.best()
        best_genome = SystemGenome(tuple(Program(data) for data in best.programs))
        best_genome.validate(shape)
        _specialized_best, best_specialization = template.specialize(
            FlatGenomeBatch.from_genomes((best_genome,), shape)
        )
    cpu_best_mse = score_setting(
        best_genome.programs,
        best.constants,
        best.bindings,
        reference,
        arguments.steps,
        shape,
    )
    materialized_source = generate_packed_cuda_module(
        FED_BATCH_MODEL,
        shape,
        arguments.kernels,
        dispatch,
        winner_output=False,
    )
    materialized_compilation = compile_cuda(materialized_source, arguments.arch, arguments.nvcc)
    materialized_inspection = inspect_module(materialized_source, materialized_compilation.cubin)
    if not isinstance(materialized_inspection.plan, PackedModulePlan):
        raise RuntimeError("materialized best-score check requires a packed module plan")
    materialized_specialization = specialize_packed_module_cubin(
        materialized_compilation.cubin,
        materialized_inspection.plan,
        (best_genome,),
        shape,
    )
    materialized_population = PackedPopulation(
        1,
        1,
        array("f", best.constants),
        array("I", best.bindings),
    )
    materialized_run = run_packed_module_fedbatch(
        materialized_specialization.cubin,
        materialized_inspection.plan,
        materialized_population,
        reference,
        shape,
        arguments.steps,
        arguments.threads,
        1,
        arguments.device,
    )
    materialized_gpu_mse = float(materialized_run.mse[0])
    gpu_invalid = not math.isfinite(best.mse) or best.mse >= 0.99 * INVALID_MSE
    materialized_invalid = not math.isfinite(materialized_gpu_mse) or materialized_gpu_mse >= 0.99 * INVALID_MSE
    cpu_invalid = not math.isfinite(cpu_best_mse) or cpu_best_mse >= 0.99 * INVALID_MSE
    correctness = {
        "hashed_gpu_mse": best.mse,
        "materialized_gpu_mse": materialized_gpu_mse,
        "cpu_mse": cpu_best_mse,
        "hashed_materialized_invalid_status_match": gpu_invalid == materialized_invalid,
        "hashed_materialized_absolute_error": None if gpu_invalid or materialized_invalid else abs(best.mse - materialized_gpu_mse),
        "cpu_invalid_status_match": materialized_invalid == cpu_invalid,
        "cpu_absolute_error": None if materialized_invalid or cpu_invalid else abs(materialized_gpu_mse - cpu_best_mse),
        "cpu_relative_error": None if materialized_invalid or cpu_invalid else abs(materialized_gpu_mse - cpu_best_mse) / max(abs(cpu_best_mse), 1.0e-30),
    }
    report = {
        "schema": "secant.system_id.c99_gp_run.v1",
        "architecture": f"sm_{inspection.plan.architecture}",
        "compile_seconds": compilation.elapsed_seconds,
        "module_bytes": len(compilation.cubin),
        "module_genome_capacity": module_capacity,
        "population_size": config.population_size,
        "settings_per_genome": config.settings_per_genome,
        "generations": arguments.generations,
        "batches_per_generation": int(stats.batches_per_generation),
        "workers": arguments.workers,
        "maximum_loaded_modules": arguments.loaded_modules,
        "threads_per_block": arguments.threads,
        "genomes_per_cta": arguments.genomes_per_cta,
        "max_nodes_per_ast": config.max_nodes_per_ast,
        "initial_max_nodes": config.initial_max_nodes,
        "timing": {
            "wall_seconds": stats.wall_seconds,
            "winner_materialization_seconds": stats.winner_materialization_seconds,
            "upload_seconds": stats.upload_seconds,
            "pipeline_seconds": stats.pipeline_seconds,
            "download_seconds": stats.download_seconds,
            "evolution_seconds": stats.evolution_seconds,
        },
        "throughput": {
            "evaluated_genomes": int(stats.evaluated_genomes),
            "evaluated_configurations": int(stats.evaluated_configurations),
            "genomes_per_second": stats.genomes_per_second,
            "configurations_per_second": stats.configurations_per_second,
        },
        "best": {
            "mse": best.mse,
            "objective": best.objective,
            "generation": best.generation,
            "complexity": best.complexity,
            "constants": list(best.constants),
            "bindings": list(best.bindings),
            "program_hex": [program.hex() for program in best.programs],
            "specialized_register_count": int(best_specialization.maximum_register_count),
            "specialized_sass_instructions": int(best_specialization.sass_instruction_count),
        },
        "correctness": correctness,
    }
    _write(build / "c99_gp_run.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _recovery_gp_config(arguments: argparse.Namespace, seed: int) -> GPConfig:
    return GPConfig(
        population_size=arguments.population,
        settings_per_genome=arguments.settings,
        max_nodes_per_ast=arguments.max_nodes,
        max_program_bytes_per_ast=arguments.max_program_bytes,
        initial_max_nodes=arguments.initial_max_nodes,
        max_depth=arguments.max_depth,
        tournament_size=arguments.tournament,
        elite_count=arguments.elites,
        seed=seed,
        subtree_crossover_probability=arguments.subtree_crossover,
        whole_site_crossover_probability=arguments.whole_site_crossover,
        subtree_mutation_probability=arguments.subtree_mutation,
        point_mutation_probability=arguments.point_mutation,
        binding_keep_probability=arguments.binding_keep,
        constant_mutation_scale=arguments.constant_scale,
        parsimony_coefficient=arguments.parsimony,
        minimum_valid_mse=0.0 if arguments.accept_exact_zero else 1.0e-30,
    )


def _first_recovery_milestone(rows: list[dict[str, object]], field: str, threshold: float) -> dict[str, object] | None:
    for row in rows:
        value = row[field]
        if isinstance(value, (float, int)) and math.isfinite(float(value)) and float(value) <= threshold:
            return {
                "generation": row["generation"],
                "elapsed_seconds": row["elapsed_seconds"],
                "value": value,
            }
    return None


def _raw_checkpoint_rows(checkpoints) -> list[dict[str, object]]:
    return [
        {
            "generation": checkpoint.generation,
            "elapsed_seconds": checkpoint.elapsed_seconds,
            "generation_best_mse": checkpoint.generation_best_mse,
            "generation_best_objective": checkpoint.generation_best_objective,
            "best_mse": checkpoint.best_mse,
            "best_objective": checkpoint.best_objective,
            "best_generation": checkpoint.best_generation,
            "best_complexity": checkpoint.best_complexity,
            "program_hex": [program.hex() for program in checkpoint.programs],
            "constants": list(checkpoint.constants),
            "bindings": list(checkpoint.bindings),
        }
        for checkpoint in checkpoints
    ]


def _add_gp_stats(total: CGPRunStats, value: CGPRunStats) -> None:
    for field in (
        "generations",
        "evaluated_genomes",
        "evaluated_configurations",
    ):
        setattr(total, field, getattr(total, field) + getattr(value, field))
    for field in (
        "winner_materialization_seconds",
        "upload_seconds",
        "pipeline_seconds",
        "download_seconds",
        "evolution_seconds",
    ):
        setattr(total, field, getattr(total, field) + getattr(value, field))
    total.batches_per_generation = value.batches_per_generation


def _promote_gp_candidates(
    gp,
    lm_template_cubin: bytes,
    lm_queue: TrajectoryLMQueue,
    arguments: argparse.Namespace,
    seed: int,
    seen_structure_ids: set[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidates = gp.candidates()
    selection = select_lm_candidates(
        candidates,
        seen_structure_ids,
        arguments.lm_selection_mode,
        arguments.lm_promotion_count,
        arguments.lm_novelty_count,
        arguments.lm_random_count,
        arguments.lm_selection_pool,
        seed,
        gp.generation,
        arguments.lm_random_trigger_probability,
    )
    seen_structure_ids.update(selection.observed_structure_ids)
    boundary = selection.boundary_record()
    valid_mse = [
        float(candidate.mse)
        for candidate in candidates
        if math.isfinite(float(candidate.mse))
        and 0.0 <= float(candidate.mse) < 0.99 * INVALID_MSE
    ]
    complexities = [int(candidate.complexity) for candidate in candidates]
    boundary["gp_population_snapshot"] = {
        "candidates": len(candidates),
        "valid_mse_candidates": len(valid_mse),
        "invalid_mse_candidates": len(candidates) - len(valid_mse),
        "best_valid_mse": min(valid_mse) if valid_mse else None,
        "median_valid_mse": statistics.median(valid_mse) if valid_mse else None,
        "worst_valid_mse": max(valid_mse) if valid_mse else None,
        "minimum_complexity": min(complexities) if complexities else None,
        "median_complexity": statistics.median(complexities) if complexities else None,
        "maximum_complexity": max(complexities) if complexities else None,
    }
    prepared = []
    skipped = []
    for selected in selection.selected:
        candidate = selected.candidate
        genome = SystemGenome(tuple(Program(data) for data in candidate.programs))
        ast_node_counts = [
            sum(
                instruction.kind != InstructionType.RETURN_F32
                for instruction in program.instructions()
            )
            for program in genome.programs
        ]
        try:
            specialization, _bundle = specialize_split_lm_cubin(
                lm_template_cubin, genome, FED_BATCH_SHAPE
            )
        except (DerivativeTapeError, SassAssemblyError) as error:
            print(
                f"skipping LM promotion for generation {gp.generation} "
                f"candidate {candidate.index} complexity {candidate.complexity} "
                f"AST nodes {ast_node_counts}: {error}",
                flush=True,
            )
            skipped.append(
                {
                    "candidate_index": candidate.index,
                    "candidate_complexity": candidate.complexity,
                    "ast_node_counts": ast_node_counts,
                    "program_hex": [program.data.hex() for program in genome.programs],
                    "structure_sha256": selected.structure_sha256,
                    "reason": str(error),
                }
            )
            continue
        population = make_lm_population(
            candidate,
            arguments.lm_settings,
            arguments.lm_starts,
            seed ^ (gp.generation << 20) ^ candidate.index,
            arguments.lm_binding_keep,
            arguments.lm_start_scale,
        )
        prepared.append((selected, specialization, population, ast_node_counts))
    boundary.update(
        {
            "eligible_union": len(prepared),
            "skipped_count": len(skipped),
            "skipped": skipped,
        }
    )
    if not prepared:
        return [], boundary
    batch = lm_queue.run(
        tuple(item[1].cubin for item in prepared),
        tuple(item[2] for item in prepared),
        arguments.steps,
        arguments.lm_iterations,
        arguments.lm_damping_attempts,
        arguments.lm_initial_damping,
        arguments.lm_threads,
    )
    boundary["lm_batch"] = {
        "candidate_count": len(prepared),
        "fits": sum(
            population.num_fits
            for _selected, _specialization, population, _ast_node_counts in prepared
        ),
        "wall_seconds": batch.wall_seconds,
        "upload_seconds": batch.upload_seconds,
        "download_seconds": batch.download_seconds,
        "sum_ticket_seconds": sum(batch.ticket_total_seconds),
        "sum_queue_wait_seconds": sum(batch.queue_wait_seconds),
        "sum_image_copy_seconds": sum(batch.image_copy_seconds),
        "sum_module_load_seconds": sum(batch.module_load_seconds),
    }
    events: list[dict[str, object]] = []
    for (selected, specialization, population, ast_node_counts), result in zip(
        prepared, batch.results
    ):
        candidate = selected.candidate
        constants, bindings = promoted_values(population, result)
        accepted = gp.candidate_improve(
            candidate.index, result.best_mse, constants, bindings
        )
        finite_fits = sum(math.isfinite(value) for value in result.mse)
        fits_with_accepted_steps = sum(value > 0 for value in result.accepted_steps)
        events.append(
            {
                "generation": gp.generation,
                "candidate_index": candidate.index,
                "candidate_complexity": candidate.complexity,
                "ast_node_counts": ast_node_counts,
                "structure_sha256": selected.structure_sha256,
                "structure_novel": selected.structure_novel,
                "selected_by_novelty": selected.selected_by_novelty,
                "selected_by_random": selected.selected_by_random,
                "selected_by_novelty_only": selected.selected_by_novelty and not selected.selected_by_random,
                "selected_by_random_only": selected.selected_by_random and not selected.selected_by_novelty,
                "selected_by_both": selected.selected_by_novelty and selected.selected_by_random,
                "before_mse": candidate.mse,
                "after_mse": result.best_mse,
                "accepted": accepted,
                "best_setting": result.best_fit // population.starts_per_setting,
                "best_start": result.best_fit % population.starts_per_setting,
                "fits": population.num_fits,
                "finite_fits": finite_fits,
                "invalid_mse_fits": population.num_fits - finite_fits,
                "total_lm_iterations": sum(result.iterations),
                "total_accepted_steps": sum(result.accepted_steps),
                "fits_with_accepted_steps": fits_with_accepted_steps,
                "fits_hitting_iteration_limit": sum(
                    value >= arguments.lm_iterations for value in result.iterations
                ),
                "best_fit_iterations": int(result.iterations[result.best_fit]),
                "best_fit_accepted_steps": int(result.accepted_steps[result.best_fit]),
                "promotion_batch_candidates": len(prepared),
                "promotion_batch_wall_seconds": batch.wall_seconds,
                "promotion_batch_upload_seconds": batch.upload_seconds,
                "promotion_batch_download_seconds": batch.download_seconds,
                "ticket_total_seconds": result.seconds,
                "module_load_seconds": result.module_load_seconds,
                "register_count": specialization.register_count,
                "primal_pressure_fallback_groups": list(
                    specialization.primal_pressure_fallback_groups
                ),
                "partial_pressure_fallback_groups": list(
                    specialization.partial_pressure_fallback_groups
                ),
                "primal_sass_instructions": list(specialization.primal_instruction_counts),
                "partial_sass_instructions": list(specialization.partial_instruction_counts),
            }
        )
    return events, boundary


def _run_gp_with_lm_promotions(
    gp,
    pipeline,
    reference_device: int,
    lm_template_cubin: bytes,
    lm_queue: TrajectoryLMQueue,
    arguments: argparse.Namespace,
    seed: int,
) -> tuple[
    CGPRunStats,
    tuple[GPCheckpoint, ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    total = CGPRunStats()
    checkpoints: list[GPCheckpoint] = []
    events: list[dict[str, object]] = []
    boundaries: list[dict[str, object]] = []
    seen_structure_ids: set[str] = set()
    started = time.perf_counter()
    completed = 0
    while completed < arguments.generations:
        chunk = min(arguments.lm_promotion_interval, arguments.generations - completed)
        chunk_started = time.perf_counter() - started
        stats, chunk_checkpoints = gp.run_traced(
            pipeline,
            chunk,
            reference_device,
            arguments.steps,
            arguments.threads,
            arguments.reduction_threads,
            1,
            arguments.full_mse_gp,
        )
        _add_gp_stats(total, stats)
        adjusted = [
            GPCheckpoint(
                chunk_started + checkpoint.elapsed_seconds,
                checkpoint.generation_best_mse,
                checkpoint.generation_best_objective,
                checkpoint.best_mse,
                checkpoint.best_objective,
                checkpoint.generation,
                checkpoint.best_generation,
                checkpoint.best_complexity,
                checkpoint.programs,
                checkpoint.constants,
                checkpoint.bindings,
            )
            for checkpoint in chunk_checkpoints
        ]
        new_events, boundary = _promote_gp_candidates(
            gp, lm_template_cubin, lm_queue, arguments, seed, seen_structure_ids
        )
        events.extend(new_events)
        boundaries.append(boundary)
        live_diagnostics = lm_run_diagnostics(boundaries, events)
        live_payload = {
            "schema": "secant.system_id.trajectory_lm_diagnostics.v1",
            "status": "running",
            "seed": seed,
            "generation": gp.generation,
            "diagnostics": live_diagnostics,
        }
        live_text = json.dumps(live_payload, indent=2) + "\n"
        diagnostics_build = Path(arguments.build_directory)
        _write(diagnostics_build / "lm_diagnostics.live.json", live_text)
        _write(diagnostics_build / f"seed_{seed}" / "lm_diagnostics.json", live_text)
        specialization_diagnostics = live_diagnostics["specialization"]
        optimization_diagnostics = live_diagnostics["optimization"]
        fallback_diagnostics = live_diagnostics["pressure_fallback"]
        print(
            f"LM diagnostics through generation {gp.generation}: "
            f"accepted {optimization_diagnostics['accepted_candidates']}/"
            f"{optimization_diagnostics['evaluated_candidates']} evaluated; "
            f"rejected {specialization_diagnostics['rejected']}/"
            f"{specialization_diagnostics['attempted']} selected; "
            f"pressure fallback "
            f"{fallback_diagnostics['evaluated_candidates_using_fallback']}",
            flush=True,
        )
        generation_best = min(
            gp.candidates(), key=lambda candidate: (candidate.objective, candidate.index)
        )
        best = gp.best()
        adjusted[-1] = GPCheckpoint(
            time.perf_counter() - started,
            generation_best.mse,
            generation_best.objective,
            best.mse,
            best.objective,
            generation_best.generation,
            best.generation,
            best.complexity,
            best.programs,
            best.constants,
            best.bindings,
        )
        for checkpoint in adjusted:
            if (
                checkpoint.generation % arguments.checkpoint_stride == 0
                or checkpoint.generation + 1 == arguments.generations
            ):
                checkpoints.append(checkpoint)
        completed += chunk
        if completed < arguments.generations:
            advance_started = time.perf_counter()
            gp.advance()
            total.evolution_seconds += time.perf_counter() - advance_started
    total.wall_seconds = time.perf_counter() - started
    total.genomes_per_second = total.evaluated_genomes / total.wall_seconds
    total.configurations_per_second = total.evaluated_configurations / total.wall_seconds
    best = gp.best()
    total.best_mse = best.mse
    total.best_objective = best.objective
    total.best_generation = best.generation
    total.best_complexity = best.complexity
    return total, tuple(checkpoints), tuple(events), tuple(boundaries)


def command_c99_recovery_run(arguments: argparse.Namespace) -> None:
    if arguments.model != "fed_batch":
        raise ValueError("fedbatch-c99-recovery-run only accepts --model fed_batch")
    if len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if arguments.materialized_gp and not arguments.full_mse_gp:
        raise ValueError("--materialized-gp requires --full-mse-gp")
    requested_full_mse_gp = bool(arguments.full_mse_gp)
    requested_materialized_gp = bool(arguments.materialized_gp)
    if arguments.lm_promotion_interval < 0:
        raise ValueError("--lm-promotion-interval cannot be negative")
    shape = _shape(arguments, FED_BATCH_MODEL)
    training_designs = {
        "original3": INITIAL_STATES,
        "diverse12": DIVERSE_INITIAL_STATES,
        "product_paired16": PRODUCT_PAIRED_INITIAL_STATES,
    }
    training_initial_states = training_designs[arguments.training_design]
    if shape.trajectory_count != len(training_initial_states) or shape.observation_count != 12:
        raise ValueError(
            f"training design {arguments.training_design} requires "
            f"--trajectories {len(training_initial_states)} and --observations 12"
        )
    if arguments.lm_promotion_interval:
        if min(
            arguments.lm_promotion_count,
            arguments.lm_settings,
            arguments.lm_starts,
            arguments.lm_iterations,
            arguments.lm_damping_attempts,
            arguments.lm_threads,
            arguments.lm_workers,
            arguments.lm_loaded_modules,
        ) <= 0:
            raise ValueError("experimental trajectory-LM dimensions must be positive")
        if not 0.0 <= arguments.lm_binding_keep <= 1.0 or arguments.lm_start_scale < 0.0:
            raise ValueError("invalid experimental trajectory-LM mutation controls")
        if arguments.training_design != "product_paired16":
            raise ValueError("experimental trajectory LM requires --training-design product_paired16")
        if arguments.loss != "relative" or arguments.relative_error_floor != 1.0:
            raise ValueError("experimental trajectory LM requires relative loss with a 1.0 denominator floor")
        if arguments.constants != 8 or shape.ast_leaf_counts != (8, 8):
            raise ValueError("experimental trajectory LM requires eight constants and two eight-leaf sites")
        if arguments.lm_selection_pool < arguments.lm_promotion_count:
            raise ValueError("--lm-selection-pool must be at least --lm-promotion-count")
        if arguments.lm_selection_mode == SELECTION_NOVELTY_RANDOM:
            if min(arguments.lm_novelty_count, arguments.lm_random_count) <= 0:
                raise ValueError("novelty-random LM selection requires positive novelty and random counts")
            if arguments.lm_novelty_count + arguments.lm_random_count > arguments.lm_promotion_count:
                raise ValueError("novelty plus random LM counts cannot exceed --lm-promotion-count")
        if not 0.0 <= arguments.lm_random_trigger_probability <= 1.0:
            raise ValueError("--lm-random-trigger-probability must be in [0, 1]")
    dispatch = PackedDispatch(arguments.genome_capacity, arguments.genomes_per_cta)
    module_capacity = arguments.kernels * dispatch.genome_capacity
    if arguments.population < module_capacity:
        raise ValueError("--population must be at least one complete module")
    batch_count = (arguments.population + module_capacity - 1) // module_capacity
    build = Path(arguments.build_directory)
    relative_error_floor = arguments.relative_error_floor if arguments.loss == "relative" else None
    source = generate_packed_cuda_module(
        FED_BATCH_MODEL,
        shape,
        arguments.kernels,
        dispatch,
        winner_output=not arguments.full_mse_gp,
        hashed_settings=not arguments.materialized_gp,
        constant_mutation_scale=arguments.constant_scale,
        binding_keep_probability=arguments.binding_keep,
        relative_error_floor=relative_error_floor,
        reject_exact_zero_score=not arguments.accept_exact_zero,
    )
    _write(build / "fedbatch_recovery_template.cu", source)
    compilation = compile_cuda(source, arguments.arch, arguments.nvcc)
    _write(build / "fedbatch_recovery_template.cubin", compilation.cubin)
    inspection = inspect_module(source, compilation.cubin)
    if not isinstance(inspection.plan, PackedModulePlan):
        raise RuntimeError("controlled recovery requires a packed module plan")
    lm_template_cubin = None
    lm_compile_seconds = 0.0
    if arguments.lm_promotion_interval:
        lm_source = render_trajectory_lm_source()
        _write(build / "trajectory_lm_template.cu", lm_source)
        lm_compilation = compile_cuda(lm_source, arguments.arch, arguments.nvcc)
        lm_template_cubin = lm_compilation.cubin
        lm_compile_seconds = lm_compilation.elapsed_seconds
        _write(build / "trajectory_lm_template.cubin", lm_template_cubin)
    unary_operations = (
        (InstructionType.NEG_F32, InstructionType.ABS_F32)
        if arguments.allow_unary
        else ()
    )
    binary_operations = [InstructionType.ADD_F32]
    if arguments.allow_subtraction:
        binary_operations.append(InstructionType.SUB_F32)
    binary_operations.append(InstructionType.MUL_F32)
    if not arguments.no_division:
        binary_operations.append(InstructionType.DIV_F32)
    binary_operations = tuple(binary_operations)
    api = C99Library(arguments.c99_library)
    reference = reference_data(
        arguments.steps,
        training_initial_states,
        shape.observation_count,
    )
    raw_runs = []
    searched_seeds: list[int] = []
    latest_progress: dict[str, object] | None = None
    progress_path = build / "progress.json"
    campaign_started = time.perf_counter()
    lm_queue = None
    try:
        if lm_template_cubin is not None:
            lm_queue = TrajectoryLMQueue(
                api,
                len(lm_template_cubin),
                reference,
                arguments.lm_promotion_count,
                arguments.lm_settings,
                arguments.lm_starts,
                arguments.lm_workers,
                arguments.lm_loaded_modules,
                arguments.device,
            )
        with api.template(
            compilation.cubin,
            inspection.plan,
            shape,
            settings_mode=SSID_SETTINGS_MATERIALIZED if arguments.materialized_gp else SSID_SETTINGS_HASHED_INCUMBENT,
        ) as template:
            with template.pipeline(
                arguments.workers,
                max(batch_count, arguments.loaded_modules),
                arguments.loaded_modules,
                arguments.device,
                enable_cuda=True,
            ) as pipeline:
                reference_device = pipeline.allocate(len(reference) * reference.itemsize)
                try:
                    pipeline.upload(reference_device, reference)
                    for seed in arguments.seeds:
                        progress = {
                            "schema": "secant.system_id.fedbatch_recovery_progress.v1",
                            "status": "running",
                            "searched_seeds": searched_seeds,
                            "verified_seeds": [],
                            "current_seed": seed,
                            "total_seeds": len(arguments.seeds),
                            "elapsed_seconds": time.perf_counter() - campaign_started,
                        }
                        if latest_progress is not None:
                            progress["latest"] = latest_progress
                        _write(progress_path, json.dumps(progress, indent=2) + "\n")
                        print(
                            f"starting recovery seed {seed} "
                            f"({len(searched_seeds) + 1}/{len(arguments.seeds)})",
                            flush=True,
                        )
                        config = _recovery_gp_config(arguments, seed)
                        with template.gp(config, shape, unary_operations, binary_operations) as gp:
                            if lm_template_cubin is None:
                                stats, checkpoints = gp.run_traced(
                                    pipeline,
                                    arguments.generations,
                                    reference_device,
                                    arguments.steps,
                                    arguments.threads,
                                    arguments.reduction_threads,
                                    arguments.checkpoint_stride,
                                    arguments.full_mse_gp,
                                )
                                lm_events = ()
                                lm_boundaries = ()
                            else:
                                assert lm_queue is not None
                                stats, checkpoints, lm_events, lm_boundaries = _run_gp_with_lm_promotions(
                                    gp,
                                    pipeline,
                                    reference_device,
                                    lm_template_cubin,
                                    lm_queue,
                                    arguments,
                                    seed,
                                )
                            best = gp.best()
                        best_genome = SystemGenome(tuple(Program(data) for data in best.programs))
                        best_genome.validate(shape)
                        _specialized, specialization = template.specialize(
                            FlatGenomeBatch.from_genomes((best_genome,), shape)
                        )
                        raw_runs.append(
                            (seed, config, stats, checkpoints, best, specialization, lm_events, lm_boundaries)
                        )
                        raw_checkpoint_path = build / f"seed_{seed}" / "checkpoints.raw.jsonl"
                        _write(
                            raw_checkpoint_path,
                            "".join(
                                json.dumps(row, separators=(",", ":")) + "\n"
                                for row in _raw_checkpoint_rows(checkpoints)
                            ),
                        )
                        searched_seeds.append(seed)
                        seed_lm_diagnostics = lm_run_diagnostics(lm_boundaries, lm_events)
                        completed_lm_diagnostics = json.dumps(
                            {
                                "schema": "secant.system_id.trajectory_lm_diagnostics.v1",
                                "status": "complete",
                                "seed": seed,
                                "generation": checkpoints[-1].generation,
                                "diagnostics": seed_lm_diagnostics,
                            },
                            indent=2,
                        ) + "\n"
                        _write(build / "lm_diagnostics.live.json", completed_lm_diagnostics)
                        _write(
                            build / f"seed_{seed}" / "lm_diagnostics.json",
                            completed_lm_diagnostics,
                        )
                        latest_progress = {
                            "seed": seed,
                            "search_seconds": stats.wall_seconds,
                            "best_mse": best.mse,
                            "best_generation": best.generation,
                            "verification_status": "pending_materialized_replay",
                            "lm_diagnostics": seed_lm_diagnostics,
                            "checkpoint_file": str(raw_checkpoint_path.relative_to(build)),
                        }
                        _write(
                            progress_path,
                            json.dumps(
                                {
                                    "schema": "secant.system_id.fedbatch_recovery_progress.v1",
                                    "status": "running" if len(searched_seeds) < len(arguments.seeds) else "analyzing",
                                    "searched_seeds": searched_seeds,
                                    "verified_seeds": [],
                                    "current_seed": None,
                                    "total_seeds": len(arguments.seeds),
                                    "elapsed_seconds": time.perf_counter() - campaign_started,
                                    "latest": latest_progress,
                                },
                                indent=2,
                            )
                            + "\n",
                        )
                        print(
                            f"finished recovery search seed {seed} in {stats.wall_seconds:.3f} s; "
                            f"provisional best MSE {best.mse:.9g}; materialized replay pending",
                            flush=True,
                        )
                finally:
                    pipeline.free(reference_device)
    finally:
        if lm_queue is not None:
            lm_queue.close()
    gpu_campaign_seconds = time.perf_counter() - campaign_started

    seed_reports: list[dict[str, object]] = []
    final_genomes: list[SystemGenome] = []
    analysis_started = time.perf_counter()
    for seed, config, stats, checkpoints, best, specialization, lm_events, lm_boundaries in raw_runs:
        checkpoint_rows: list[dict[str, object]] = []
        for checkpoint in checkpoints:
            programs = tuple(Program(data) for data in checkpoint.programs)
            genome = SystemGenome(programs)
            genome.validate(shape)
            rate_metrics = rate_surface_metrics(
                programs,
                checkpoint.constants,
                checkpoint.bindings,
                shape,
            )
            checkpoint_rows.append(
                {
                    "generation": checkpoint.generation,
                    "elapsed_seconds": checkpoint.elapsed_seconds,
                    "generation_best_mse": checkpoint.generation_best_mse,
                    "best_mse": checkpoint.best_mse,
                    "best_objective": checkpoint.best_objective,
                    "best_generation": checkpoint.best_generation,
                    "best_complexity": checkpoint.best_complexity,
                    "strict_structure_match": strict_structure_match(programs, checkpoint.bindings, shape),
                    "site_structure_match": list(site_structure_matches(programs, checkpoint.bindings, shape)),
                    "rate_surface_nrmse": rate_metrics.joint_nrmse,
                    "rate_site_nrmse": list(rate_metrics.site_nrmse),
                    "worst_site_rate_nrmse": max(rate_metrics.site_nrmse),
                    "program_hex": [program.data.hex() for program in programs],
                    "constants": list(checkpoint.constants),
                    "bindings": list(checkpoint.bindings),
                }
            )
        checkpoint_path = build / f"seed_{seed}" / "checkpoints.jsonl"
        _write(checkpoint_path, "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in checkpoint_rows))
        best_genome = SystemGenome(tuple(Program(data) for data in best.programs))
        final_genomes.append(best_genome)
        final_rates = rate_surface_metrics(best_genome.programs, best.constants, best.bindings, shape)
        final_training_cpu = score_setting(
            best_genome.programs,
            best.constants,
            best.bindings,
            reference,
            arguments.steps,
            shape,
            relative_error_floor,
        )
        final_held_out = held_out_trajectory_mse(
            best_genome.programs,
            best.constants,
            best.bindings,
            shape,
            steps_per_observation=arguments.steps,
        )
        strict_rows = [row for row in checkpoint_rows if row["strict_structure_match"]]
        lm_diagnostics = lm_run_diagnostics(lm_boundaries, lm_events)
        lm_wall_seconds = float(lm_diagnostics["timing"]["wall_seconds"])
        measured_stage_seconds = {
            "gp_winner_materialization": stats.winner_materialization_seconds,
            "gp_upload": stats.upload_seconds,
            "gp_pipeline": stats.pipeline_seconds,
            "gp_download": stats.download_seconds,
            "gp_evolution": stats.evolution_seconds,
            "trajectory_lm_batches": lm_wall_seconds,
        }
        accounted_seconds = sum(measured_stage_seconds.values())
        seed_reports.append(
            {
                "seed": seed,
                "search": {
                    "wall_seconds": stats.wall_seconds,
                    "genomes_per_second": stats.genomes_per_second,
                    "configurations_per_second": stats.configurations_per_second,
                    "evaluated_genomes": int(stats.evaluated_genomes),
                    "evaluated_configurations": int(stats.evaluated_configurations),
                    "measured_stage_seconds": measured_stage_seconds,
                    "measured_stage_fraction_of_wall": {
                        name: seconds / stats.wall_seconds if stats.wall_seconds > 0.0 else None
                        for name, seconds in measured_stage_seconds.items()
                    },
                    "unattributed_control_seconds": stats.wall_seconds - accounted_seconds,
                },
                "lm_promotions": list(lm_events),
                "lm_selection_boundaries": list(lm_boundaries),
                "lm_selection_summary": lm_selection_summary(lm_boundaries, lm_events),
                "lm_diagnostics": lm_diagnostics,
                "milestones": {
                    "training_mse": {
                        format(threshold, ".9g"): _first_recovery_milestone(checkpoint_rows, "best_mse", threshold)
                        for threshold in arguments.training_thresholds
                    },
                    "worst_site_rate_nrmse": {
                        format(threshold, ".9g"): _first_recovery_milestone(checkpoint_rows, "worst_site_rate_nrmse", threshold)
                        for threshold in arguments.rate_thresholds
                    },
                    "strict_structure": None
                    if not strict_rows
                    else {
                        "generation": strict_rows[0]["generation"],
                        "elapsed_seconds": strict_rows[0]["elapsed_seconds"],
                    },
                },
                "best": {
                    "gpu_training_mse": best.mse,
                    "cpu_training_mse": final_training_cpu,
                    "held_out_trajectory_mse": final_held_out,
                    "rate_surface_nrmse": final_rates.joint_nrmse,
                    "rate_site_nrmse": list(final_rates.site_nrmse),
                    "worst_site_rate_nrmse": max(final_rates.site_nrmse),
                    "strict_structure_match": strict_structure_match(best_genome.programs, best.bindings, shape),
                    "site_structure_match": list(site_structure_matches(best_genome.programs, best.bindings, shape)),
                    "generation": best.generation,
                    "complexity": best.complexity,
                    "expressions": [
                        resolved_expression(program, best.constants, best.bindings, shape)
                        for program in best_genome.programs
                    ],
                    "program_hex": [program.data.hex() for program in best_genome.programs],
                    "constants": list(best.constants),
                    "bindings": list(best.bindings),
                    "specialized_register_count": int(specialization.maximum_register_count),
                    "specialized_sass_instructions": int(specialization.sass_instruction_count),
                },
                "checkpoint_file": str(checkpoint_path.relative_to(build)),
            }
        )
    analysis_seconds = time.perf_counter() - analysis_started

    materialized_source = generate_packed_cuda_module(
        FED_BATCH_MODEL,
        shape,
        arguments.kernels,
        dispatch,
        winner_output=False,
        relative_error_floor=relative_error_floor,
        reject_exact_zero_score=not arguments.accept_exact_zero,
    )
    materialized_compilation = compile_cuda(materialized_source, arguments.arch, arguments.nvcc)
    materialized_inspection = inspect_module(materialized_source, materialized_compilation.cubin)
    if not isinstance(materialized_inspection.plan, PackedModulePlan):
        raise RuntimeError("materialized recovery replay requires a packed module plan")
    for report, genome, (
        _seed,
        _config,
        _stats,
        _checkpoints,
        best,
        _specialization,
        _lm_events,
        _lm_boundaries,
    ) in zip(seed_reports, final_genomes, raw_runs):
        specialized = specialize_packed_module_cubin(
            materialized_compilation.cubin,
            materialized_inspection.plan,
            (genome,),
            shape,
        )
        population = PackedPopulation(1, 1, array("f", best.constants), array("I", best.bindings))
        replay = run_packed_module_fedbatch(
            specialized.cubin,
            materialized_inspection.plan,
            population,
            reference,
            shape,
            arguments.steps,
            arguments.threads,
            1,
            arguments.device,
        )
        materialized_mse = float(replay.mse[0])
        report["best"]["materialized_gpu_mse"] = materialized_mse
        replay_error, replay_tolerance, replay_passed = _materialized_replay_verdict(
            float(best.mse), materialized_mse
        )
        report["best"]["search_replay_absolute_error"] = replay_error
        report["best"]["search_replay_tolerance"] = replay_tolerance
        report["best"]["search_replay_passed"] = replay_passed
        if not arguments.materialized_gp:
            report["best"]["hashed_materialized_absolute_error"] = replay_error

    replay_failures = [
        int(seed["seed"])
        for seed in seed_reports
        if not bool(seed["best"]["search_replay_passed"])
    ]

    threshold_summary: dict[str, object] = {}
    for category, thresholds in (
        ("training_mse", arguments.training_thresholds),
        ("worst_site_rate_nrmse", arguments.rate_thresholds),
    ):
        category_summary = {}
        for threshold in thresholds:
            key = format(threshold, ".9g")
            successful = [
                report["milestones"][category][key]
                for report in seed_reports
                if report["milestones"][category][key] is not None
            ]
            category_summary[key] = {
                "successful_seeds": len(successful),
                "total_seeds": len(seed_reports),
                "median_generation_on_success": None
                if not successful
                else statistics.median(float(item["generation"]) for item in successful),
                "median_seconds_on_success": None
                if not successful
                else statistics.median(float(item["elapsed_seconds"]) for item in successful),
            }
        threshold_summary[category] = category_summary
    strict_successes = [report for report in seed_reports if report["milestones"]["strict_structure"] is not None]
    campaign_lm_boundaries = [
        {**boundary, "seed": seed}
        for seed, _config, _stats, _checkpoints, _best, _specialization, _events, boundaries in raw_runs
        for boundary in boundaries
    ]
    campaign_lm_events = [
        {**event, "seed": seed}
        for seed, _config, _stats, _checkpoints, _best, _specialization, events, _boundaries in raw_runs
        for event in events
    ]
    campaign_lm_diagnostics = lm_run_diagnostics(campaign_lm_boundaries, campaign_lm_events)
    held_out_summary = {}
    for threshold in arguments.held_out_thresholds:
        successful = [
            seed for seed in seed_reports
            if math.isfinite(float(seed["best"]["held_out_trajectory_mse"]))
            and float(seed["best"]["held_out_trajectory_mse"]) <= threshold
        ]
        held_out_summary[format(threshold, ".9g")] = {
            "successful_seeds": len(successful),
            "total_seeds": len(seed_reports),
        }
    report = {
        "schema": "secant.system_id.fedbatch_recovery.v1",
        "experiment": "both fed-batch rate laws blinded; known mass-balance equations fixed",
        "training_design": {
            "name": arguments.training_design,
            "trajectory_count": len(training_initial_states),
            "initial_states": [list(state) for state in training_initial_states],
        },
        "architecture": f"sm_{inspection.plan.architecture}",
        "compile_seconds": compilation.elapsed_seconds,
        "trajectory_lm_compile_seconds": lm_compile_seconds,
        "materialized_replay_compile_seconds": materialized_compilation.elapsed_seconds,
        "module_bytes": len(compilation.cubin),
        "population_size": arguments.population,
        "settings_per_genome": arguments.settings,
        "generations": arguments.generations,
        "seeds": list(arguments.seeds),
        "grammar": {
            "binary": [operation.name for operation in binary_operations],
            "unary": [operation.name for operation in unary_operations],
            "max_nodes_per_ast": arguments.max_nodes,
            "initial_max_nodes": arguments.initial_max_nodes,
            "max_depth": arguments.max_depth,
            "parsimony_coefficient": arguments.parsimony,
        },
        "loss": {
            "kind": "relative_state_mse" if relative_error_floor is not None else "raw_state_mse",
            "denominator_floor": relative_error_floor,
            "reject_exact_zero_score": not arguments.accept_exact_zero,
        },
        "checkpoint_stride": arguments.checkpoint_stride,
        "gp_score_output": "full_mse_cpu_argmin" if arguments.full_mse_gp else "fused_gpu_winner",
        "gp_settings_mode": "materialized" if arguments.materialized_gp else "hashed_incumbent",
        "correctness": {
            "result_validity": "valid" if not replay_failures else "invalid",
            "all_final_winners_passed_materialized_replay": not replay_failures,
            "failed_replay_seeds": replay_failures,
            "requested_score_output": "full_mse_cpu_argmin" if requested_full_mse_gp else "fused_gpu_winner",
            "requested_settings_mode": "materialized" if requested_materialized_gp else "hashed_incumbent",
            "fallback_applied": False,
            "fallback_reason": None,
        },
        "trajectory_lm": {
            "enabled": bool(arguments.lm_promotion_interval),
            "promotion_interval_generations": arguments.lm_promotion_interval,
            "candidates_per_promotion": arguments.lm_promotion_count,
            "selection_mode": arguments.lm_selection_mode,
            "structure_definition": "normalized_postorder_operator_and_leaf_alias_pattern_v1",
            "selection_pool_unique_structures": arguments.lm_selection_pool,
            "novelty_candidates_per_boundary": arguments.lm_novelty_count,
            "random_candidates_per_boundary": arguments.lm_random_count,
            "random_trigger_probability": arguments.lm_random_trigger_probability,
            "binding_settings_per_candidate": arguments.lm_settings,
            "starts_per_setting": arguments.lm_starts,
            "iterations": arguments.lm_iterations,
            "damping_attempts": arguments.lm_damping_attempts,
            "threads_per_block": arguments.lm_threads,
            "loader_workers": arguments.lm_workers,
            "maximum_loaded_modules": arguments.lm_loaded_modules,
        },
        "trajectory_lm_diagnostics": campaign_lm_diagnostics,
        "timing": {
            "gpu_campaign_and_control_seconds": gpu_campaign_seconds,
            "offline_checkpoint_analysis_seconds": analysis_seconds,
            "sum_search_seconds": sum(float(seed["search"]["wall_seconds"]) for seed in seed_reports),
            "median_search_seconds": statistics.median(float(seed["search"]["wall_seconds"]) for seed in seed_reports),
        },
        "success_summary": {
            **threshold_summary,
            "final_held_out_trajectory_mse": held_out_summary,
            "strict_structure": {
                "successful_seeds": len(strict_successes),
                "total_seeds": len(seed_reports),
            },
        },
        "seed_results": seed_reports,
    }
    _write(build / "recovery_report.json", json.dumps(report, indent=2) + "\n")
    _write(
        progress_path,
        json.dumps(
            {
                "schema": "secant.system_id.fedbatch_recovery_progress.v1",
                "status": "complete" if not replay_failures else "failed_correctness",
                "searched_seeds": searched_seeds,
                "verified_seeds": [] if replay_failures else searched_seeds,
                "completed_seeds": [] if replay_failures else searched_seeds,
                "current_seed": None,
                "total_seeds": len(arguments.seeds),
                "elapsed_seconds": time.perf_counter() - campaign_started,
                "report": "recovery_report.json",
            },
            indent=2,
        )
        + "\n",
    )
    print(json.dumps(report, indent=2))
    if replay_failures:
        raise RuntimeError(
            "materialized replay rejected recovery results for seeds "
            + ", ".join(str(seed) for seed in replay_failures)
        )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="fed_batch", help="fed_batch or Python module:attribute")
    parser.add_argument("--constants", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--observations", type=int, default=12)
    parser.add_argument("--patch-capacity", type=int, default=192)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="secant-system-id")
    subparsers = parser.add_subparsers(required=True)

    generate = subparsers.add_parser("generate", help="generate the CUDA template")
    _common(generate)
    generate.add_argument("-o", "--output", default="-", help="output CUDA path, or - for stdout")
    generate.set_defaults(function=command_generate)

    compile_parser = subparsers.add_parser("compile", help="compile and inspect a CUDA template")
    _common(compile_parser)
    compile_parser.add_argument("--arch", default="sm_120")
    compile_parser.add_argument("--nvcc")
    compile_parser.add_argument("-o", "--output", default="generated/fedbatch_template.cubin")
    compile_parser.set_defaults(function=command_compile)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a generated CUBIN")
    inspect_parser.add_argument("cubin")
    inspect_parser.add_argument("--kernel", default="ssid_score")
    inspect_parser.add_argument("-o", "--output", default="-", help="output JSON path, or - for stdout")
    inspect_parser.set_defaults(function=command_inspect)

    specialize = subparsers.add_parser("specialize", help="write planted post-order ASTs into a template")
    specialize.add_argument("cubin")
    specialize.add_argument("-o", "--output", default="generated/fedbatch_specialized.cubin")
    specialize.set_defaults(function=command_specialize)

    run = subparsers.add_parser("fedbatch-run", help="compile, specialize, load, and benchmark the first kernel")
    _common(run)
    run.add_argument("--arch", default="sm_120")
    run.add_argument("--nvcc")
    run.add_argument("--settings", type=int, default=16384)
    run.add_argument("--steps", type=int, default=16)
    run.add_argument("--threads", type=int, default=128)
    run.add_argument("--repetitions", type=int, default=10)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--build-directory", default="generated")
    run.set_defaults(function=command_run)

    packed_run = subparsers.add_parser(
        "fedbatch-packed-run",
        help="compile, pack, and benchmark multiple system genomes per CTA",
    )
    _common(packed_run)
    packed_run.add_argument("--arch", default="sm_120")
    packed_run.add_argument("--nvcc")
    packed_run.add_argument("--genomes", type=int, default=8)
    packed_run.add_argument("--genome-capacity", type=int, default=8)
    packed_run.add_argument("--genomes-per-cta", type=int, default=8)
    packed_run.add_argument("--settings", type=int, default=2048)
    packed_run.add_argument("--steps", type=int, default=16)
    packed_run.add_argument("--threads", type=int, default=128)
    packed_run.add_argument("--repetitions", type=int, default=10)
    packed_run.add_argument("--seed", type=int, default=7)
    packed_run.add_argument("--device", type=int, default=0)
    packed_run.add_argument(
        "--cpu-check-all-settings",
        action="store_true",
        help="CPU-score every genome and setting instead of only setting zero",
    )
    packed_run.add_argument("--build-directory", default="generated")
    packed_run.set_defaults(function=command_packed_run)

    module_generate = subparsers.add_parser(
        "packed-module-generate",
        help="generate one CUDA module containing several packed kernel entry points",
    )
    _common(module_generate)
    module_generate.add_argument("--kernels", type=int, default=2)
    module_generate.add_argument("--genome-capacity", type=int, default=8)
    module_generate.add_argument("--genomes-per-cta", type=int, default=8)
    module_generate.add_argument("-o", "--output", default="-")
    module_generate.set_defaults(function=command_packed_module_generate)

    module_run = subparsers.add_parser(
        "fedbatch-module-run",
        help="compile, specialize, load once, and run several packed kernel entry points",
    )
    _common(module_run)
    module_run.add_argument("--arch", default="sm_120")
    module_run.add_argument("--nvcc")
    module_run.add_argument("--kernels", type=int, default=2)
    module_run.add_argument("--genomes", type=int, default=16)
    module_run.add_argument("--genome-capacity", type=int, default=8)
    module_run.add_argument("--genomes-per-cta", type=int, default=8)
    module_run.add_argument("--settings", type=int, default=2048)
    module_run.add_argument("--steps", type=int, default=16)
    module_run.add_argument("--threads", type=int, default=128)
    module_run.add_argument("--repetitions", type=int, default=10)
    module_run.add_argument("--seed", type=int, default=7)
    module_run.add_argument("--device", type=int, default=0)
    module_run.add_argument(
        "--cpu-check-all-settings",
        action="store_true",
        help="CPU-score every genome and setting instead of only setting zero",
    )
    module_run.add_argument("--build-directory", default="generated")
    module_run.set_defaults(function=command_packed_module_run)

    c99_run = subparsers.add_parser(
        "fedbatch-c99-pipeline-run",
        help="benchmark flat postorder batches through the C99 eager module pipeline",
    )
    _common(c99_run)
    c99_run.add_argument("--arch", default="sm_120")
    c99_run.add_argument("--nvcc")
    c99_run.add_argument("--c99-library")
    c99_run.add_argument("--kernels", type=int, default=2)
    c99_run.add_argument("--genomes", type=int, default=16)
    c99_run.add_argument("--genome-capacity", type=int, default=8)
    c99_run.add_argument("--genomes-per-cta", type=int, default=8)
    c99_run.add_argument("--settings", type=int, default=2048)
    c99_run.add_argument("--steps", type=int, default=16)
    c99_run.add_argument("--threads", type=int, default=128)
    c99_run.add_argument("--workers", type=int, default=4)
    c99_run.add_argument("--worker-counts", type=_worker_counts, default=(1, 2, 4, 8))
    c99_run.add_argument("--specialization-submissions", type=int, default=5000)
    c99_run.add_argument("--submissions", type=int, default=32)
    c99_run.add_argument("--queue-capacity", type=int, default=64)
    c99_run.add_argument("--loaded-modules", type=int, default=8)
    c99_run.add_argument("--seed", type=int, default=7)
    c99_run.add_argument("--device", type=int, default=0)
    c99_run.add_argument("--cpu-check-all-settings", action="store_true")
    c99_run.add_argument("--build-directory", default="generated")
    c99_run.set_defaults(function=command_c99_pipeline_run)

    unique_run = subparsers.add_parser(
        "fedbatch-c99-unique-sweep",
        help="benchmark byte-distinct genome modules across eager depths and settings",
    )
    _common(unique_run)
    unique_run.add_argument("--arch", default="sm_120")
    unique_run.add_argument("--nvcc")
    unique_run.add_argument("--c99-library")
    unique_run.add_argument("--kernels", type=int, default=1)
    unique_run.add_argument("--genomes", type=int, default=128)
    unique_run.add_argument("--genome-capacity", type=int, default=128)
    unique_run.add_argument("--genomes-per-cta", type=int, default=1)
    unique_run.add_argument(
        "--warp-per-system",
        action="store_true",
        help="assign one system to each warp; experimental and limited to 32 settings",
    )
    unique_run.add_argument("--settings-counts", type=_worker_counts, default=(4096, 8192, 16384))
    unique_run.add_argument("--loaded-module-depths", type=_worker_counts, default=(1, 2, 3, 4))
    unique_run.add_argument(
        "--execution-stream-counts",
        type=_worker_counts,
        default=(1,),
        help="same-context CUDA stream counts to benchmark for each loaded-module depth",
    )
    unique_run.add_argument("--steps", type=int, default=16)
    unique_run.add_argument("--threads", type=int, default=256)
    unique_run.add_argument("--workers", type=int, default=1)
    unique_run.add_argument("--submissions", type=int, default=512)
    unique_run.add_argument("--winner-output", action="store_true")
    unique_run.add_argument(
        "--structurally-diverse",
        action="store_true",
        help="use deterministic random AST structures and reject duplicate systems or site ASTs",
    )
    unique_run.add_argument("--reduction-threads", type=int, default=256)
    unique_run.add_argument("--seed", type=int, default=7)
    unique_run.add_argument("--device", type=int, default=0)
    unique_run.add_argument("--build-directory", default="generated/unique_pipeline")
    unique_run.set_defaults(function=command_c99_unique_pipeline_sweep)

    gp_run = subparsers.add_parser(
        "fedbatch-c99-gp-run",
        help="run the fully C99 system-genome GP loop through the eager CUDA pipeline",
    )
    _common(gp_run)
    gp_run.add_argument("--arch", default="sm_120")
    gp_run.add_argument("--nvcc")
    gp_run.add_argument("--c99-library")
    gp_run.add_argument("--kernels", type=int, default=1)
    gp_run.add_argument("--genome-capacity", type=int, default=128)
    gp_run.add_argument("--genomes-per-cta", type=int, default=1)
    gp_run.add_argument("--population", type=int, default=512)
    gp_run.add_argument("--settings", type=int, default=512)
    gp_run.add_argument("--generations", type=int, default=50)
    gp_run.add_argument("--steps", type=int, default=16)
    gp_run.add_argument("--threads", type=int, default=256)
    gp_run.add_argument("--reduction-threads", type=int, default=256)
    gp_run.add_argument("--workers", type=int, default=1)
    gp_run.add_argument("--loaded-modules", type=int, default=2)
    gp_run.add_argument("--max-nodes", type=int, default=30)
    gp_run.add_argument("--max-program-bytes", type=int, default=192)
    gp_run.add_argument("--initial-max-nodes", type=int, default=15)
    gp_run.add_argument("--max-depth", type=int, default=12)
    gp_run.add_argument("--tournament", type=int, default=5)
    gp_run.add_argument("--elites", type=int, default=8)
    gp_run.add_argument("--subtree-crossover", type=float, default=0.35)
    gp_run.add_argument("--whole-site-crossover", type=float, default=0.15)
    gp_run.add_argument("--subtree-mutation", type=float, default=0.25)
    gp_run.add_argument("--point-mutation", type=float, default=0.15)
    gp_run.add_argument("--binding-keep", type=float, default=0.5)
    gp_run.add_argument("--constant-scale", type=float, default=0.5)
    gp_run.add_argument("--parsimony", type=float, default=0.0)
    gp_run.add_argument("--no-unary", action="store_true")
    gp_run.add_argument("--seed", type=int, default=7)
    gp_run.add_argument("--device", type=int, default=0)
    gp_run.add_argument("--build-directory", default="generated/c99_gp")
    gp_run.set_defaults(function=command_c99_gp_run)

    recovery_run = subparsers.add_parser(
        "fedbatch-c99-recovery-run",
        help="run controlled multi-seed recovery of both blinded fed-batch rate laws",
    )
    _common(recovery_run)
    recovery_run.add_argument("--arch", default="sm_120")
    recovery_run.add_argument("--nvcc")
    recovery_run.add_argument("--c99-library")
    recovery_run.add_argument("--kernels", type=int, default=1)
    recovery_run.add_argument("--genome-capacity", type=int, default=128)
    recovery_run.add_argument("--genomes-per-cta", type=int, default=1)
    recovery_run.add_argument("--population", type=int, default=1024)
    recovery_run.add_argument("--settings", type=int, default=512)
    recovery_run.add_argument("--generations", type=int, default=500)
    recovery_run.add_argument("--steps", type=int, default=16)
    recovery_run.add_argument("--threads", type=int, default=256)
    recovery_run.add_argument("--reduction-threads", type=int, default=256)
    recovery_run.add_argument("--workers", type=int, default=1)
    recovery_run.add_argument("--loaded-modules", type=int, default=2)
    recovery_run.add_argument("--max-nodes", type=int, default=30)
    recovery_run.add_argument("--max-program-bytes", type=int, default=192)
    recovery_run.add_argument("--initial-max-nodes", type=int, default=15)
    recovery_run.add_argument("--max-depth", type=int, default=12)
    recovery_run.add_argument("--tournament", type=int, default=5)
    recovery_run.add_argument("--elites", type=int, default=8)
    recovery_run.add_argument("--subtree-crossover", type=float, default=0.35)
    recovery_run.add_argument("--whole-site-crossover", type=float, default=0.15)
    recovery_run.add_argument("--subtree-mutation", type=float, default=0.25)
    recovery_run.add_argument("--point-mutation", type=float, default=0.15)
    recovery_run.add_argument("--binding-keep", type=float, default=0.5)
    recovery_run.add_argument("--constant-scale", type=float, default=0.5)
    recovery_run.add_argument("--relative-error-floor", type=float, default=1.0)
    recovery_run.add_argument("--loss", choices=("raw", "relative"), default="relative")
    recovery_run.add_argument("--checkpoint-stride", type=int, default=1)
    recovery_run.add_argument(
        "--full-mse-gp",
        action="store_true",
        help="write every setting score and select per-genome winners in C",
    )
    recovery_run.add_argument(
        "--materialized-gp",
        action="store_true",
        help="materialize settings and bindings in C before each full-MSE launch",
    )
    recovery_run.add_argument(
        "--accept-exact-zero",
        action="store_true",
        help="accept an exact zero accumulated SSE instead of treating it as a corrupted score",
    )
    recovery_run.add_argument(
        "--training-design",
        choices=("original3", "diverse12", "product_paired16"),
        default="original3",
    )
    recovery_run.add_argument("--parsimony", type=float, default=1.0e-5)
    recovery_run.add_argument("--allow-unary", action="store_true")
    recovery_run.add_argument("--allow-subtraction", action="store_true")
    recovery_run.add_argument(
        "--no-division",
        action="store_true",
        help="exclude division from the recovery GP grammar",
    )
    recovery_run.add_argument(
        "--lm-promotion-interval",
        type=int,
        default=0,
        help="experimentally run trajectory LM on the best eligible genomes every N generations; zero disables it",
    )
    recovery_run.add_argument("--lm-promotion-count", type=int, default=1)
    recovery_run.add_argument(
        "--lm-selection-mode",
        choices=(SELECTION_OBJECTIVE, SELECTION_NOVELTY_RANDOM, SELECTION_RANDOM),
        default=SELECTION_OBJECTIVE,
        help="select LM candidates by objective or compare structural novelty with a random control",
    )
    recovery_run.add_argument("--lm-selection-pool", type=int, default=32)
    recovery_run.add_argument("--lm-novelty-count", type=int, default=2)
    recovery_run.add_argument("--lm-random-count", type=int, default=2)
    recovery_run.add_argument("--lm-random-trigger-probability", type=float, default=0.14)
    recovery_run.add_argument("--lm-settings", type=int, default=8192)
    recovery_run.add_argument("--lm-starts", type=int, default=4)
    recovery_run.add_argument("--lm-binding-keep", type=float, default=0.75)
    recovery_run.add_argument("--lm-start-scale", type=float, default=0.5)
    recovery_run.add_argument("--lm-iterations", type=int, default=20)
    recovery_run.add_argument("--lm-damping-attempts", type=int, default=8)
    recovery_run.add_argument("--lm-initial-damping", type=float, default=1.0e-3)
    recovery_run.add_argument("--lm-threads", type=int, default=128)
    recovery_run.add_argument("--lm-workers", type=int, default=2)
    recovery_run.add_argument("--lm-loaded-modules", type=int, default=4)
    recovery_run.add_argument("--seeds", type=_worker_counts, default=(7, 17, 29, 43, 61))
    recovery_run.add_argument(
        "--training-thresholds",
        type=_positive_float_counts,
        default=(0.1, 0.01, 0.001, 0.0001),
    )
    recovery_run.add_argument(
        "--rate-thresholds",
        type=_positive_float_counts,
        default=(0.1, 0.03, 0.01),
    )
    recovery_run.add_argument(
        "--held-out-thresholds",
        type=_positive_float_counts,
        default=(0.1, 0.01, 0.001),
    )
    recovery_run.add_argument("--device", type=int, default=0)
    recovery_run.add_argument("--build-directory", default="generated/fedbatch_recovery")
    recovery_run.set_defaults(function=command_c99_recovery_run)

    arguments = parser.parse_args(argv)
    arguments.function(arguments)


if __name__ == "__main__":
    main()
