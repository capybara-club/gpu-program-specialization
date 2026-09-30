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

from .compiler import compile_cuda
from .fed_batch import FED_BATCH_MODEL, FED_BATCH_SHAPE, reference_data
from .toggle_benchmark import _correctness_samples
from .toggle_fed_batch import (
    benchmark_diverse_toggle_systems,
    make_toggle_population,
)
from .toggle_module_runtime import run_toggle_cuda_module
from .toggle_template import generate_toggle_cuda_module


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure compiled-unique AST throughput in a resident multi-entry CUDA module."
    )
    parser.add_argument("--arch", default="sm_120")
    parser.add_argument("--nvcc")
    parser.add_argument("--systems", type=int, default=256)
    parser.add_argument("--systems-per-kernel", type=int, default=16)
    parser.add_argument("--constant-banks", type=int, default=1)
    parser.add_argument(
        "--configurations",
        type=int,
        default=0,
        help="Evaluate only the first N available toggle configurations (0 means all).",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--checks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0x5EC4A7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-directory", default="generated/toggle_unique")
    arguments = parser.parse_args(argv)
    if arguments.systems <= 0 or arguments.systems_per_kernel <= 0:
        raise ValueError("system counts must be positive")
    if arguments.systems % arguments.systems_per_kernel:
        raise ValueError("--systems must divide evenly by --systems-per-kernel")
    if arguments.constant_banks <= 0:
        raise ValueError("--constant-banks must be positive")

    systems = benchmark_diverse_toggle_systems(
        arguments.systems,
        FED_BATCH_SHAPE,
        arguments.seed,
    )
    structural_signatures = {
        tuple(int(instruction.kind) for instruction in program.instructions())
        for system in systems
        for program in system.programs
    }
    population = make_toggle_population(systems, arguments.constant_banks)
    available_configurations = population.configurations_per_system
    configurations = arguments.configurations or available_configurations
    if not 0 < configurations <= available_configurations:
        raise ValueError("--configurations must fit in the generated toggle population")
    threads = arguments.threads or min(512, configurations)
    module = generate_toggle_cuda_module(
        FED_BATCH_MODEL,
        FED_BATCH_SHAPE,
        systems,
        arguments.systems_per_kernel,
        population.toggle_bit_count,
    )
    build = Path(arguments.build_directory)
    build.mkdir(parents=True, exist_ok=True)
    (build / "toggle_unique_module.cu").write_text(module.source)
    compilation = compile_cuda(module.source, arguments.arch, arguments.nvcc)
    (build / "toggle_unique_module.cubin").write_bytes(compilation.cubin)
    (build / "compile.log").write_text(compilation.log)
    reference = reference_data(arguments.steps)
    run = run_toggle_cuda_module(
        compilation.cubin,
        module.kernel_names,
        module.systems_per_kernel,
        population,
        reference,
        FED_BATCH_SHAPE,
        arguments.steps,
        threads,
        arguments.repetitions,
        arguments.streams,
        arguments.device,
        configurations,
    )
    failures, samples = _correctness_samples(
        systems,
        population,
        run.mse,
        reference,
        arguments.steps,
        arguments.checks,
        configurations,
    )
    seconds = run.seconds_per_sweep
    system_count = len(systems)
    component_ast_count = sum(len(system.programs) for system in systems)
    toggle_structures = system_count * population.permutations_per_bank
    evaluated_configurations = system_count * configurations
    report = {
        "schema": "secant.system_id.toggle_unique_benchmark.v1",
        "scope": "resident multi-entry CUDA module sweep; compilation and module loading separate",
        "architecture": arguments.arch,
        "system_family": "structurally distinct synthetic fixed-cost postorder ASTs",
        "compiled_unique_systems": system_count,
        "compiled_unique_component_asts": component_ast_count,
        "unique_component_structure_signatures": len(structural_signatures),
        "systems_per_kernel": arguments.systems_per_kernel,
        "kernel_count": len(module.kernel_names),
        "toggle_bits": population.toggle_bit_count,
        "toggle_structures_per_system": population.permutations_per_bank,
        "constant_banks_per_system": population.constant_bank_count,
        "available_configurations_per_system": available_configurations,
        "configurations_per_system": configurations,
        "evaluated_configurations_per_sweep": evaluated_configurations,
        "threads_per_block": threads,
        "execution_streams": run.execution_streams,
        "steps_per_observation": arguments.steps,
        "source_bytes": len(module.source.encode()),
        "cubin_bytes": len(compilation.cubin),
        "compile_seconds": compilation.elapsed_seconds,
        "module_load_seconds": run.module_load_seconds,
        "function_lookup_seconds": run.function_lookup_seconds,
        "maximum_registers_per_thread": run.maximum_registers_per_thread,
        "maximum_local_bytes_per_thread": run.maximum_local_bytes_per_thread,
        "seconds_per_module_sweep": seconds,
        "compiled_unique_systems_per_second": system_count / seconds,
        "compiled_unique_component_asts_per_second": component_ast_count / seconds,
        "unique_toggle_structures_per_second": toggle_structures / seconds,
        "evaluated_configurations_per_second": evaluated_configurations / seconds,
        "nanoseconds_per_evaluated_configuration": seconds * 1.0e9 / evaluated_configurations,
        "cpu_correctness": {
            "checked": len(samples),
            "failures": failures,
            "reference_precision": "Python float64 reference compared with CUDA FP32 rollout",
            "samples": samples,
        },
        "deviations": [
            "This is resident execution of one already-loaded multi-entry CUDA module; compile, load, lookup, transfer, and unload time are excluded from throughput.",
            "The ASTs are structurally distinct deterministic fixed-operation-count samples, not a measured GP population distribution or known ground-truth recovery set.",
            "Every system uses the same known fed-batch CUDA envelope, trajectory data, leaf count, operator multiset, and five available toggle bits for a controlled dispatch comparison.",
            "The path writes every MSE and implements only dense aligned trajectories.",
        ],
    }
    (build / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
