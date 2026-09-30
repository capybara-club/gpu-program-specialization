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
"""Run the same secant-sr policy over every built-in problem and write CSV."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import copy
import csv
import gzip
import hashlib
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time

from srbench_v2 import (
    SRBENCH_V2_BLACKBOX_DATASET_COUNT,
    SRBENCH_V2_BLACKBOX_MANIFEST_SHA256,
    SRBENCH_V2_BLACKBOX_PROTOCOL,
    SRBENCH_V2_FEYNMAN_DATASET_COUNT,
    SRBENCH_V2_FEYNMAN_MANIFEST_SHA256,
    SRBENCH_V2_PROTOCOL,
    SRBENCH_V2_SEEDS,
    SRBENCH_V2_TARGET_NOISES,
)


DEFAULT_PROBLEMS = (
    "nguyen1",
    "nguyen5",
    "rational2",
    "distance2",
    "interaction3",
    "oscillator2",
)

FEYNMAN_SUPPORTED_PROBLEMS = (
    "feynman_I_12_1",
    "feynman_I_14_3",
    "feynman_I_14_4",
    "feynman_I_16_6",
    "feynman_I_18_12",
    "feynman_I_47_23",
    "feynman_II_8_31",
    "feynman_II_38_14",
    "feynman_III_15_12",
    "feynman_III_17_37",
)

FEYNMAN_QD_PANEL_PROBLEMS = (
    "feynman_III_12_43",
    "feynman_II_11_3",
    "feynman_I_13_12",
    "feynman_I_18_4",
    "feynman_I_40_1",
    "feynman_I_44_4",
    "feynman_test_12",
    "feynman_III_4_32",
    "feynman_I_6_2a",
    "feynman_III_15_14",
    "feynman_I_34_14",
    "feynman_I_48_2",
    "feynman_II_13_23",
    "feynman_II_13_34",
    "feynman_I_15_10",
    "feynman_II_8_7",
    "feynman_III_19_51",
    "feynman_III_8_54",
    "feynman_I_9_18",
    "feynman_test_1",
    "feynman_test_4",
    "feynman_test_9",
    "feynman_I_50_26",
    "feynman_II_35_21",
)

SRBENCH_BLACKBOX_SMOKE_PROBLEMS = (
    "1027_ESL",
    "1029_LEV",
    "201_pol",
    "227_cpu_small",
    "503_wind",
    "505_tecator",
)

SECANT_SR_MAX_INPUTS = 128
SECANT_SR_MAX_INLINE_ASM_OPERANDS = 256
SECANT_SR_MIXED_CONTROL_REGISTER_RESERVE = 32


def dynamic_leaf_fallback_reason(
    num_inputs: int,
    num_dynamic_leaves: int,
    asts_per_kernel: int,
) -> str | None:
    """Explain why a black-box problem cannot use the mixed static-input kernel."""
    if num_inputs + num_dynamic_leaves > SECANT_SR_MAX_INPUTS:
        return f"static_plus_dynamic_inputs_exceed_{SECANT_SR_MAX_INPUTS}"
    num_sources = num_inputs + num_dynamic_leaves + 1  # One target.
    num_inline_operands = 2 * num_sources + asts_per_kernel + 1  # Sources, SSE outputs, keepalive.
    required_registers = num_inline_operands + SECANT_SR_MIXED_CONTROL_REGISTER_RESERVE
    if required_registers > SECANT_SR_MAX_INLINE_ASM_OPERANDS:
        return f"mixed_kernel_register_budget_exceeds_{SECANT_SR_MAX_INLINE_ASM_OPERANDS}"
    return None

PROBLEM_SUITES = {
    "core": DEFAULT_PROBLEMS,
    "feynman-supported": FEYNMAN_SUPPORTED_PROBLEMS,
    "feynman-qd-panel": FEYNMAN_QD_PANEL_PROBLEMS,
    "feynman-all": (),
    "srbench-blackbox-all": (),
    "srbench-blackbox-smoke": SRBENCH_BLACKBOX_SMOKE_PROBLEMS,
}

TEMPLATE_PREFIX = "template_generated "
CACHE_PREFIX = "template_cache "
MSE_REDUCER_CACHE_PREFIX = "mse_reducer_cache "
MATURITY_STAGE_PREFIX = "maturity_stage "
CHECKPOINT_PREFIX = "checkpoint "
TEMPLATE_RECIPE_FIELDS = (
    "secant_version", "recipe_version", "recipe_flags", "cc_major", "cc_minor", "nvrtc_major", "nvrtc_minor",
    "ptxas_opt_level", "nvrtc_no_cache", "shape", "shape_value", "num_kernels",
    "asts_per_kernel", "num_inputs", "num_input_columns", "num_static_input_columns", "num_input_constants",
    "num_dynamic_leaves",
    "num_targets", "tile_rows", "threads_per_block", "patch_capacity_instructions",
)
TEMPLATE_EVENT_FIELDS = (
    "problem", "seed", "event_index", "template_key", *TEMPLATE_RECIPE_FIELDS, "source_bytes", "cubin_bytes",
)
MATURITY_STAGE_FIELDS = (
    "problem", "seed", "generation", "population",
    "exploratory_before", "resolved_before", "mixed_refined_before", "constant_refined_before",
    "full_selected", "full_promoted", "full_finite_gain_count", "full_nonfinite_recovered", "full_promotion_rate",
    "full_r2_gain_mean", "full_r2_gain_max", "full_seconds",
    "mixed_eligible", "mixed_selected", "mixed_promoted", "mixed_finite_gain_count", "mixed_nonfinite_recovered",
    "mixed_promotion_rate", "mixed_r2_gain_mean", "mixed_r2_gain_max", "mixed_seconds",
    "constant_eligible", "constant_selected", "constant_promoted", "constant_finite_gain_count",
    "constant_nonfinite_recovered", "constant_promotion_rate", "constant_r2_gain_mean", "constant_r2_gain_max",
    "constant_seconds", "static_seconds",
    "exploratory_after", "resolved_after", "mixed_refined_after", "constant_refined_after",
    "leaf_settings", "leaf_setting_policy", "leaf_setting_update_seconds",
    "mixed_holes", "mixed_probability", "constant_optimizer", "constant_settings",
    "lm_bindings", "lm_starts_per_binding", "lm_column_binding_winners", "lm_column_binding_promoted",
    "constant_iterations",
    "constant_budget", "constant_interval", "constant_random_fraction",
    "best_r2_after_static", "best_r2_after_full", "best_r2_after_mixed", "best_r2_after_constant",
    "best_sse_after_static", "best_sse_after_full", "best_sse_after_mixed", "best_sse_after_constant",
    "best_origin", "best_maturity",
    "train_r2", "validation_r2", "complexity", "depth", "evaluation_seconds", "elapsed_seconds",
)
CHECKPOINT_FIELDS = (
    "system", "problem", "seed", "generation", "search_elapsed_seconds", "train_r2", "validation_r2",
    "validation_seconds", "complexity", "nodes", "fingerprint", "program_bytes", "checkpoint_copy_seconds",
    "program_hex",
)


def parse_generation(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split())


def template_record_parse(line: str, problem: str, seed: int, event_index: int) -> dict[str, str | int]:
    values = parse_generation(line[len(TEMPLATE_PREFIX):])
    identity = {field: values.get(field, "") for field in TEMPLATE_RECIPE_FIELDS}
    template_key = "|".join(identity[field] for field in TEMPLATE_RECIPE_FIELDS)
    return {
        "problem": problem,
        "seed": seed,
        "event_index": event_index,
        "template_key": template_key,
        **values,
    }


def prepared_dataset_get(args: argparse.Namespace, problem: str, seed: int):
    if args.pmlb_root is not None:
        from srbench_data import (
            prepare_dataset,
            prepare_srbench_v2_blackbox_dataset,
            prepare_srbench_v2_groundtruth_dataset,
        )

        source = args.pmlb_root / "datasets" / problem / f"{problem}.tsv.gz"
        if args.protocol == SRBENCH_V2_PROTOCOL:
            return prepare_srbench_v2_groundtruth_dataset(
                source,
                args.prepared_data_dir,
                seed,
                args.target_noise,
            )
        if args.protocol == SRBENCH_V2_BLACKBOX_PROTOCOL:
            return prepare_srbench_v2_blackbox_dataset(
                source,
                args.prepared_data_dir,
                seed,
            )
        return prepare_dataset(
            source,
            args.prepared_data_dir,
            seed,
            args.rows,
            args.validation_rows,
            args.scale_x,
            args.scale_y,
        )
    return None


def problem_args_get(args: argparse.Namespace, problem: str) -> argparse.Namespace:
    if problem not in args.dynamic_leaf_fallback_problems:
        return args
    effective = copy.copy(args)
    effective.backend = "cubin-dynamic-leaf"
    effective.constant_optimizer = "legacy"
    effective.constant_optimizer_iterations = 1
    effective.constant_settings = 0
    effective.constant_optimize_budget = 0
    effective.constant_optimize_probability = None
    return effective


def search_command_create(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.executable),
        "--backend", args.backend,
        "--population", str(args.population),
        "--generations", str(args.generations),
        "--workers", str(args.workers),
        "--streams", str(args.streams),
        "--leaf-settings", str(args.leaf_settings),
        "--leaf-setting-policy", args.leaf_setting_policy,
        "--constant-settings", str(args.constant_settings),
        "--constant-optimize-budget", str(args.constant_optimize_budget),
        "--constant-optimize-interval", str(args.constant_optimize_interval),
        "--constant-optimize-random-fraction", str(args.constant_optimize_random_fraction),
        "--constant-optimizer", args.constant_optimizer,
        "--constant-optimizer-iterations", str(args.constant_optimizer_iterations),
        "--constant-optimizer-scale", str(args.constant_optimizer_scale),
        "--constant-optimizer-decay", str(args.constant_optimizer_decay),
        "--constant-sweep-phase", args.constant_sweep_phase,
        "--lm-batch-asts", str(args.lm_batch_asts),
        "--lm-settings-per-cta", str(args.lm_settings_per_cta),
        "--lm-starts-per-binding", str(args.lm_starts_per_binding),
        "--lm-tile-rows", str(args.lm_tile_rows),
        "--lm-threads", str(args.lm_threads),
        "--lm-patch-instructions", str(args.lm_patch_instructions),
        "--lm-initial-damping", str(args.lm_initial_damping),
        "--dynamic-leaves", str(args.dynamic_leaves),
        "--mixed-dynamic-leaves", str(args.mixed_dynamic_leaves),
        "--mixed-refine-probability", str(args.mixed_refine_probability),
        "--dynamic-max-nodes", str(args.dynamic_max_nodes),
        "--dynamic-generations", str(args.dynamic_generations),
        "--kernels", str(args.kernels),
        "--asts-per-kernel", str(args.asts_per_kernel),
        "--static-tile-rows", str(args.static_tile_rows),
        "--patch-instructions-per-ast", str(args.patch_instructions_per_ast),
        "--final-cpu-optimize", str(args.final_cpu_optimize),
        "--final-cpu-optimizer-rows", str(args.final_cpu_optimizer_rows),
        "--final-cpu-optimizer-iterations", str(args.final_cpu_optimizer_iterations),
        "--final-cpu-optimizer-restarts", str(args.final_cpu_optimizer_restarts),
        "--final-cpu-optimizer-f-calls-limit", str(args.final_cpu_optimizer_f_calls_limit),
        "--operator-profile", args.operator_profile,
        "--qd-column-buckets", str(args.qd_column_buckets),
        "--qd-transcendental-buckets", str(args.qd_transcendental_buckets),
        "--archive-parent-probability", str(args.archive_parent_probability),
        "--constant-setting-credit-weight", str(args.constant_setting_credit_weight),
        "--parsimony-coefficient", str(args.parsimony_coefficient),
        "--validation-mode", args.validation_mode,
        "--stop-metric", args.stop_metric,
        "--stop-r2", str(args.stop_r2),
        "--time-limit-seconds", str(args.time_limit_seconds),
    ]
    if args.constant_optimize_probability is not None:
        command.extend(("--constant-optimize-probability", str(args.constant_optimize_probability)))
    if args.tile_rows != 0:
        command.extend(("--tile-rows", str(args.tile_rows)))
    if args.backend.startswith("cubin") and not args.no_cubin_cache:
        command.extend(("--cubin-cache", str(args.cubin_cache)))
    return command


def search_environment_create(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    if args.backend.startswith("cubin"):
        environment["CUDA_MODULE_LOADING"] = "EAGER"
    if args.template_output is not None:
        environment["SECANT_SR_TEMPLATE_TRACE"] = "1"
    if args.generation_output is not None:
        environment["SECANT_SR_MATURITY_TRACE"] = "1"
    if args.checkpoint_output is not None:
        environment["SECANT_SR_CHECKPOINT_TRACE"] = "1"
    return environment


def search_output_parse(
    args: argparse.Namespace,
    problem: str,
    seed: int,
    prepared,
    output: str,
    elapsed: float,
) -> tuple[
    dict[str, str | int | float],
    list[dict[str, str | int]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    generation: dict[str, str] | None = None
    expression = ""
    templates: list[dict[str, str | int]] = []
    maturity_stages: list[dict[str, str]] = []
    checkpoints: list[dict[str, str]] = []
    cache_values: dict[str, str] | None = None
    mse_reducer_cache_values: dict[str, str] | None = None
    final_optimizer_values: dict[str, str] | None = None
    process_setup_seconds = 0.0

    for line in output.splitlines():
        if line.startswith("process_setup "):
            values = parse_generation(line[len("process_setup "):])
            process_setup_seconds = (
                float(values.get("context_create_seconds", "0"))
                + float(values.get("cache_open_seconds", "0"))
            )
        elif line.startswith(f"problem={problem} generation="):
            generation = parse_generation(line)
        elif line.startswith(f"problem={problem} best_expression="):
            expression = line.split(" best_expression=", 1)[1]
        elif line.startswith(TEMPLATE_PREFIX):
            templates.append(template_record_parse(line, problem, seed, len(templates)))
        elif line.startswith(CACHE_PREFIX):
            cache_values = parse_generation(line[len(CACHE_PREFIX):])
        elif line.startswith(MSE_REDUCER_CACHE_PREFIX):
            mse_reducer_cache_values = parse_generation(line[len(MSE_REDUCER_CACHE_PREFIX):])
        elif line.startswith(MATURITY_STAGE_PREFIX):
            values = parse_generation(line[len(MATURITY_STAGE_PREFIX):])
            missing = tuple(field for field in MATURITY_STAGE_FIELDS if field not in values)
            if missing:
                raise RuntimeError(
                    f"maturity stage output for {problem}, seed {seed} is missing fields: {', '.join(missing)}"
                )
            maturity_stages.append({field: values[field] for field in MATURITY_STAGE_FIELDS})
        elif line.startswith(CHECKPOINT_PREFIX):
            values = parse_generation(line[len(CHECKPOINT_PREFIX):])
            source_fields = CHECKPOINT_FIELDS[1:]
            missing = tuple(field for field in source_fields if field not in values)
            if missing:
                raise RuntimeError(
                    f"checkpoint output for {problem}, seed {seed} is missing fields: {', '.join(missing)}"
                )
            values["search_elapsed_seconds"] = (
                f"{float(values['search_elapsed_seconds']) + process_setup_seconds:.9f}"
            )
            checkpoints.append({"system": "secant", **{field: values[field] for field in source_fields}})
        elif line.startswith("final_cpu_optimizer "):
            final_optimizer_values = parse_generation(line[len("final_cpu_optimizer "):])
    if generation is None:
        raise RuntimeError(f"no generation output for {problem}, seed {seed}\n{output}")
    if final_optimizer_values is None:
        raise RuntimeError(f"no final CPU optimizer audit for {problem}, seed {seed}\n{output}")
    validation_r2 = float(final_optimizer_values["validation_r2_after"])
    train_r2 = float(final_optimizer_values["train_r2_after"])
    symbolic = None
    if args.protocol == SRBENCH_V2_PROTOCOL and not args.skip_symbolic_assessment:
        from srbench_symbolic import assess_expression

        symbolic = assess_expression(prepared.source_path, expression, validation_r2)
    if cache_values is None:
        cache_values = {
            "enabled": "0", "hits": "0", "misses": "0", "invalidations": "0",
            "open_seconds": "0", "lookup_seconds": "0", "store_seconds": "0",
            "actual_compile_seconds": "0", "estimated_uncached_compile_seconds": "0",
            "estimated_saved_compile_seconds": "0", "prepare_seconds": "0",
        }
    if mse_reducer_cache_values is None:
        mse_reducer_cache_values = {
            "enabled": "0", "hits": "0", "misses": "0", "invalidations": "0",
            "lookup_seconds": "0", "store_seconds": "0", "actual_compile_seconds": "0",
            "estimated_uncached_compile_seconds": "0",
        }
    cache_overhead_seconds = (
        float(cache_values["open_seconds"])
        + float(cache_values["lookup_seconds"])
        + float(cache_values["store_seconds"])
        + float(mse_reducer_cache_values["lookup_seconds"])
        + float(mse_reducer_cache_values["store_seconds"])
    )
    estimated_uncached_elapsed_seconds = max(
        0.0,
        elapsed - cache_overhead_seconds
        - float(cache_values["actual_compile_seconds"])
        - float(mse_reducer_cache_values["actual_compile_seconds"])
        + float(cache_values["estimated_uncached_compile_seconds"])
        + float(mse_reducer_cache_values["estimated_uncached_compile_seconds"]),
    )
    return {
        "problem": problem,
        "seed": seed,
        "backend": args.backend,
        "generation": int(generation["generation"]),
        "train_r2": train_r2,
        "validation_r2": validation_r2,
        "complexity": int(generation["complexity"]),
        "nodes": int(generation["nodes"]),
        "depth": int(generation["depth"]),
        "unique_columns": int(generation["unique_columns"]),
        "column_occurrences": int(generation["column_occurrences"]),
        "constants": int(generation["constants"]),
        "unary_ops": int(generation["unary_ops"]),
        "binary_ops": int(generation["binary_ops"]),
        "ternary_ops": int(generation["ternary_ops"]),
        "transcendental_ops": int(generation["transcendental_ops"]),
        "constant_setting_robustness": float(generation["constant_setting_robustness"]),
        "dynamic_promoted": int(generation["dynamic_promoted"]),
        "constant_promoted": int(generation["constant_promoted"]),
        "materialize_validation_r2": float(generation["materialize_validation_r2"]),
        "materialize_validation_diverged": int(generation["materialize_validation_diverged"]),
        "archive_cells": int(generation["archive_cells"]),
        "archive_occupied_cells": int(generation["archive_occupied_cells"]),
        "archive_elites": int(generation["archive_elites"]),
        "final_cpu_optimizer_enabled": int(final_optimizer_values["enabled"]),
        "final_cpu_optimizer_constants": int(final_optimizer_values["constants"]),
        "final_cpu_optimizer_improved": int(final_optimizer_values["improved"]),
        "final_cpu_optimizer_f_calls": int(final_optimizer_values["f_calls"]),
        "final_cpu_optimizer_seconds": float(final_optimizer_values["seconds"]),
        "pre_final_train_r2": float(final_optimizer_values["train_r2_before"]),
        "pre_final_validation_r2": float(final_optimizer_values["validation_r2_before"]),
        "validation_mode": args.validation_mode,
        "stop_metric": generation["stop_metric"],
        "stop_r2": float(generation["stop_r2"]),
        "stop_value": float(generation["stop_value"]),
        "stop_reached": int(generation["stop_reached"]),
        "time_limit_seconds": float(generation["time_limit_seconds"]),
        "time_limit_reached": int(generation["time_limit_reached"]),
        "search_elapsed_seconds": float(generation["elapsed_seconds"]),
        "termination_reason": (
            "target" if int(generation["stop_reached"]) != 0 else
            "time_limit" if int(generation["time_limit_reached"]) != 0 else
            "generation_limit"
        ),
        "first_hit_seconds": (
            float(generation["elapsed_seconds"]) if int(generation["stop_reached"]) != 0 else ""
        ),
        "solved": "" if args.protocol == SRBENCH_V2_BLACKBOX_PROTOCOL else int(validation_r2 > args.stop_r2),
        "accuracy_solution": (
            "" if args.protocol == SRBENCH_V2_BLACKBOX_PROTOCOL else int(validation_r2 > 0.999)
        ),
        "symbolic_solution": int(symbolic.symbolic_solution) if symbolic is not None else "",
        "symbolic_error_is_zero": int(symbolic.symbolic_error_is_zero) if symbolic is not None else "",
        "symbolic_error_is_constant": int(symbolic.symbolic_error_is_constant) if symbolic is not None else "",
        "symbolic_fraction_is_constant": int(symbolic.symbolic_fraction_is_constant) if symbolic is not None else "",
        "simplified_expression": symbolic.simplified_expression if symbolic is not None else "",
        "symbolic_error": symbolic.symbolic_error if symbolic is not None else "",
        "symbolic_fraction": symbolic.symbolic_fraction if symbolic is not None else "",
        "symbolic_assessment_error": symbolic.assessment_error if symbolic is not None else "",
        "elapsed_seconds": elapsed,
        "estimated_uncached_elapsed_seconds": estimated_uncached_elapsed_seconds,
        "template_cache_enabled": int(cache_values["enabled"]),
        "template_cache_hits": int(cache_values["hits"]),
        "template_cache_misses": int(cache_values["misses"]),
        "template_cache_invalidations": int(cache_values["invalidations"]),
        "template_cache_open_seconds": float(cache_values["open_seconds"]),
        "template_cache_lookup_seconds": float(cache_values["lookup_seconds"]),
        "template_cache_store_seconds": float(cache_values["store_seconds"]),
        "template_actual_compile_seconds": float(cache_values["actual_compile_seconds"]),
        "template_estimated_uncached_compile_seconds": float(cache_values["estimated_uncached_compile_seconds"]),
        "template_estimated_saved_compile_seconds": float(cache_values["estimated_saved_compile_seconds"]),
        "template_prepare_seconds": float(cache_values["prepare_seconds"]),
        "mse_reducer_cache_enabled": int(mse_reducer_cache_values["enabled"]),
        "mse_reducer_cache_hits": int(mse_reducer_cache_values["hits"]),
        "mse_reducer_cache_misses": int(mse_reducer_cache_values["misses"]),
        "mse_reducer_cache_invalidations": int(mse_reducer_cache_values["invalidations"]),
        "mse_reducer_cache_lookup_seconds": float(mse_reducer_cache_values["lookup_seconds"]),
        "mse_reducer_cache_store_seconds": float(mse_reducer_cache_values["store_seconds"]),
        "mse_reducer_actual_compile_seconds": float(mse_reducer_cache_values["actual_compile_seconds"]),
        "mse_reducer_estimated_uncached_compile_seconds": float(
            mse_reducer_cache_values["estimated_uncached_compile_seconds"]
        ),
        "mse_reducer_estimated_saved_compile_seconds": max(
            0.0,
            float(mse_reducer_cache_values["estimated_uncached_compile_seconds"])
            - float(mse_reducer_cache_values["actual_compile_seconds"]),
        ),
        "best_expression": expression,
        "dataset_source": "pmlb" if prepared is not None else "generated",
        "protocol": prepared.protocol if prepared is not None else args.protocol,
        "target_noise": prepared.target_noise if prepared is not None else args.target_noise,
        "source_sha256": prepared.source_sha256 if prepared is not None else "",
        "train_rows": prepared.num_train_rows if prepared is not None else args.rows,
        "validation_rows": prepared.num_validation_rows if prepared is not None else args.validation_rows,
        "num_inputs": prepared.num_inputs if prepared is not None else "",
        "scale_x": int(prepared.scale_x) if prepared is not None else 0,
        "scale_y": int(prepared.scale_y) if prepared is not None else 0,
        "operator_profile": args.operator_profile,
        "qd_column_buckets": args.qd_column_buckets,
        "qd_transcendental_buckets": args.qd_transcendental_buckets,
        "archive_parent_probability": args.archive_parent_probability,
        "dynamic_max_nodes": args.dynamic_max_nodes,
        "dynamic_leaves": int(generation["dynamic_leaves"]),
        "leaf_settings": args.leaf_settings,
        "leaf_setting_policy": args.leaf_setting_policy,
        "mixed_dynamic_leaves": int(generation["mixed_dynamic_leaves"]),
        "mixed_refine_probability": args.mixed_refine_probability,
        "constant_settings": args.constant_settings,
        "constant_optimize_probability": args.constant_optimize_probability,
        "constant_optimize_budget": args.constant_optimize_budget,
        "constant_optimize_interval": args.constant_optimize_interval,
        "constant_optimize_random_fraction": args.constant_optimize_random_fraction,
        "constant_optimizer": args.constant_optimizer,
        "constant_optimizer_iterations": args.constant_optimizer_iterations,
        "constant_optimizer_scale": args.constant_optimizer_scale,
        "constant_optimizer_decay": args.constant_optimizer_decay,
        "constant_sweep_phase": args.constant_sweep_phase,
        "lm_batch_asts": args.lm_batch_asts,
        "lm_settings_per_cta": args.lm_settings_per_cta,
        "lm_bindings": args.constant_settings // args.lm_starts_per_binding
            if args.constant_optimizer == "lm" and args.constant_settings else 0,
        "lm_starts_per_binding": args.lm_starts_per_binding if args.constant_optimizer == "lm" else 0,
        "lm_tile_rows": args.lm_tile_rows,
        "lm_threads": args.lm_threads,
        "lm_patch_instructions": args.lm_patch_instructions,
        "lm_initial_damping": args.lm_initial_damping,
        "constant_setting_credit_weight": args.constant_setting_credit_weight,
        "parsimony_coefficient": args.parsimony_coefficient,
    }, templates, maturity_stages, checkpoints


def run_one(
    args: argparse.Namespace,
    problem: str,
    seed: int,
) -> tuple[
    dict[str, str | int | float],
    list[dict[str, str | int]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    prepared = prepared_dataset_get(args, problem, seed)
    command = search_command_create(args)

    command.extend((
        "--problem", problem,
        "--rows", str(prepared.num_train_rows if prepared is not None else args.rows),
        "--validation-rows", str(prepared.num_validation_rows if prepared is not None else args.validation_rows),
        "--seed", str(seed),
    ))
    if prepared is not None:
        command.extend(("--dataset-binary", str(prepared.path)))
    begin = time.monotonic()
    completed = subprocess.run(command, check=False, text=True, capture_output=True, env=search_environment_create(args))
    elapsed = time.monotonic() - begin
    if completed.returncode != 0:
        raise RuntimeError(
            f"search failed for {problem}, seed {seed}, exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return search_output_parse(args, problem, seed, prepared, completed.stdout, elapsed)


def run_batch(
    args: argparse.Namespace,
    jobs: list[tuple[str, int]],
) -> Iterator[tuple[
    dict[str, str | int | float],
    list[dict[str, str | int]],
    list[dict[str, str]],
    list[dict[str, str]],
]]:
    prepared_jobs = [(problem, seed, prepared_dataset_get(args, problem, seed)) for problem, seed in jobs]
    prepared_by_key = {(problem, seed): prepared for problem, seed, prepared in prepared_jobs}
    manifest_path: Path | None = None
    process: subprocess.Popen | None = None
    reader_thread: threading.Thread | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix="secant_sr_batch_",
            suffix=".tsv",
            delete=False,
        ) as manifest:
            manifest_path = Path(manifest.name)
            for problem, seed, prepared in prepared_jobs:
                dataset_path = "" if prepared is None else str(prepared.path)
                if any(character in problem or character in dataset_path for character in ("\t", "\n", "\r")):
                    raise ValueError("batch problem names and dataset paths may not contain tabs or newlines")
                rows = args.rows if prepared is None else prepared.num_train_rows
                validation_rows = args.validation_rows if prepared is None else prepared.num_validation_rows
                manifest.write(f"{problem}\t{seed}\t{dataset_path}\t{rows}\t{validation_rows}\n")
        command = search_command_create(args)
        command.extend(("--batch-manifest", str(manifest_path)))
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=stderr,
                bufsize=1,
                env=search_environment_create(args),
            )
            if process.stdout is None:
                raise RuntimeError("persistent search stdout pipe was not created")

            output_queue: queue.Queue[tuple[str, object]] = queue.Queue()

            def output_read() -> None:
                current_key: tuple[str, int] | None = None
                current_generation = ""
                current_expression = ""
                current_cache = ""
                current_mse_reducer_cache = ""
                current_final_optimizer = ""
                current_templates: list[str] = []
                current_maturity_stages: list[str] = []
                current_checkpoints: list[str] = []
                completed_keys: set[tuple[str, int]] = set()
                setup_share = 0.0

                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    if line.startswith("process_setup "):
                        values = parse_generation(line[len("process_setup "):])
                        setup_share = (
                            float(values.get("context_create_seconds", "0"))
                            + float(values.get("cache_open_seconds", "0"))
                        ) / len(jobs)
                    elif line.startswith("request_begin "):
                        values = parse_generation(line[len("request_begin "):])
                        key = (values["problem"], int(values["seed"]))
                        if current_key is not None or key not in prepared_by_key or key in completed_keys:
                            output_queue.put(("error", f"invalid persistent request start: {line}"))
                            return
                        current_key = key
                        current_generation = ""
                        current_expression = ""
                        current_cache = ""
                        current_mse_reducer_cache = ""
                        current_final_optimizer = ""
                        current_templates = []
                        current_maturity_stages = []
                        current_checkpoints = []
                    elif line.startswith("request_complete "):
                        values = parse_generation(line[len("request_complete "):])
                        key = (values["problem"], int(values["seed"]))
                        if current_key != key or int(values["status"]) != 0:
                            output_queue.put(("error", f"invalid persistent request completion: {line}"))
                            return
                        relevant_lines = [
                            f"process_setup context_create_seconds={setup_share:.9f} cache_open_seconds=0",
                            current_generation,
                            current_expression,
                            *current_templates,
                            current_cache,
                            current_mse_reducer_cache,
                            *current_maturity_stages,
                            *current_checkpoints,
                            current_final_optimizer,
                        ]
                        output_queue.put(("result", (
                            key,
                            "\n".join(item for item in relevant_lines if item),
                            float(values["elapsed_seconds"]) + setup_share,
                        )))
                        completed_keys.add(key)
                        current_key = None
                    elif current_key is not None:
                        problem, _ = current_key
                        if line.startswith(f"problem={problem} generation="):
                            current_generation = line
                        elif line.startswith(f"problem={problem} best_expression="):
                            current_expression = line
                        elif line.startswith(TEMPLATE_PREFIX):
                            current_templates.append(line)
                        elif line.startswith(CACHE_PREFIX):
                            current_cache = line
                        elif line.startswith(MSE_REDUCER_CACHE_PREFIX):
                            current_mse_reducer_cache = line
                        elif line.startswith(MATURITY_STAGE_PREFIX):
                            current_maturity_stages.append(line)
                        elif line.startswith(CHECKPOINT_PREFIX):
                            current_checkpoints.append(line)
                        elif line.startswith("final_cpu_optimizer "):
                            current_final_optimizer = line

                output_queue.put(("done", (current_key, completed_keys)))

            reader_thread = threading.Thread(target=output_read, name="secant-sr-output", daemon=True)
            reader_thread.start()
            completed_keys: set[tuple[str, int]] = set()
            current_key: tuple[str, int] | None = None
            failure = ""
            while True:
                kind, payload = output_queue.get()
                if kind == "result":
                    key, output_text, elapsed = payload
                    problem, seed = key
                    yield search_output_parse(
                        args,
                        problem,
                        seed,
                        prepared_by_key[key],
                        output_text,
                        elapsed,
                    )
                elif kind == "error":
                    failure = str(payload)
                    break
                else:
                    current_key, completed_keys = payload
                    break

            reader_thread.join()
            reader_thread = None
            process.stdout.close()
            returncode = process.wait()
            stderr.seek(0)
            stderr_output = stderr.read()
            process = None
            if failure:
                raise RuntimeError(f"{failure}\nstderr:\n{stderr_output}")
            if returncode != 0:
                raise RuntimeError(
                    f"persistent search failed with exit code {returncode}\n"
                    f"stderr:\n{stderr_output}"
                )
            if current_key is not None:
                raise RuntimeError(f"persistent request did not complete: {current_key}")
            if len(completed_keys) != len(jobs):
                missing = [key for key in jobs if key not in completed_keys]
                raise RuntimeError(f"persistent output missing requests: {missing[:8]}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if reader_thread is not None:
            reader_thread.join(timeout=5.0)
        if manifest_path is not None:
            manifest_path.unlink(missing_ok=True)


def run_pending(
    args: argparse.Namespace,
    jobs: list[tuple[str, int]],
) -> Iterator[tuple[
    tuple[str, int],
    tuple[
        dict[str, str | int | float],
        list[dict[str, str | int]],
        list[dict[str, str]],
        list[dict[str, str]],
    ],
]]:
    if not args.persistent_process:
        for problem, seed in jobs:
            job_args = problem_args_get(args, problem)
            yield (problem, seed), run_one(job_args, problem, seed)
        return

    regular = [job for job in jobs if job[0] not in args.dynamic_leaf_fallback_problems]
    fallback = [job for job in jobs if job[0] in args.dynamic_leaf_fallback_problems]
    for group in (regular, fallback):
        if not group:
            continue
        group_args = problem_args_get(args, group[0][0])
        for job, result in zip(group, run_batch(group_args, group), strict=True):
            yield job, result


def template_summary_write(
    path: Path,
    records: list[dict[str, str | int]],
) -> None:
    grouped: dict[str, list[dict[str, str | int]]] = {}

    for record in records:
        grouped.setdefault(str(record["template_key"]), []).append(record)
    fields = (
        "template_key", "occurrences", "num_problems", "problems", *TEMPLATE_RECIPE_FIELDS,
        "source_bytes", "cubin_bytes",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for template_key, occurrences in sorted(grouped.items()):
            first = occurrences[0]
            writer.writerow({
                "template_key": template_key,
                "occurrences": len(occurrences),
                "num_problems": len({str(record["problem"]) for record in occurrences}),
                "problems": ";".join(sorted({str(record["problem"]) for record in occurrences})),
                **{field: first.get(field, "") for field in TEMPLATE_RECIPE_FIELDS},
                "source_bytes": first.get("source_bytes", ""),
                "cubin_bytes": first.get("cubin_bytes", ""),
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=Path("build/secant_sr_search"))
    parser.add_argument(
        "--backend",
        choices=("cpu", "cubin", "cubin-dynamic-leaf", "cubin-staged", "cubin-maturity"),
        default="cubin",
    )
    parser.add_argument("--suite", choices=tuple(PROBLEM_SUITES), default="core")
    parser.add_argument("--problems", nargs="+", default=None)
    parser.add_argument(
        "--dataset-max-inputs",
        type=int,
        default=0,
        help="skip datasets with more input columns than this; zero disables the filter",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--population", type=int, default=8192)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--validation-rows", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--leaf-settings", type=int, default=4096)
    parser.add_argument(
        "--leaf-setting-policy",
        choices=("legacy", "legacy-rotating", "virtual-bank-fixed", "virtual-bank"),
        default="legacy",
    )
    parser.add_argument("--constant-settings", type=int, default=8192)
    parser.add_argument("--constant-optimize-budget", type=int, default=4096)
    parser.add_argument("--constant-optimize-interval", type=int, default=1)
    parser.add_argument("--constant-optimize-random-fraction", type=float, default=0.25)
    parser.add_argument("--constant-optimize-probability", type=float, default=None,
        help="legacy independent candidate sampling; overrides bounded selection when supplied")
    parser.add_argument("--constant-optimizer", choices=("legacy", "lm"), default="legacy")
    parser.add_argument("--constant-optimizer-iterations", type=int, default=1)
    parser.add_argument("--constant-optimizer-scale", type=float, default=1.0)
    parser.add_argument("--constant-optimizer-decay", type=float, default=0.5,
        help="per-iteration geometric scale multiplier; one keeps a constant radius")
    parser.add_argument("--constant-sweep-phase", choices=("each", "early", "final", "both"), default="each")
    parser.add_argument("--lm-batch-asts", type=int, default=16)
    parser.add_argument("--lm-settings-per-cta", type=int, default=128)
    parser.add_argument("--lm-starts-per-binding", type=int, default=4)
    parser.add_argument("--lm-tile-rows", type=int, default=128)
    parser.add_argument("--lm-threads", type=int, default=128)
    parser.add_argument("--lm-patch-instructions", type=int, default=2048)
    parser.add_argument("--lm-initial-damping", type=float, default=1.0e-3)
    parser.add_argument("--dynamic-leaves", type=int, default=8)
    parser.add_argument("--mixed-dynamic-leaves", type=int, default=4)
    parser.add_argument("--mixed-refine-probability", type=float, default=0.25)
    parser.add_argument("--dynamic-max-nodes", type=int, default=30)
    parser.add_argument("--dynamic-generations", type=int, default=5)
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--asts-per-kernel", type=int, default=32)
    parser.add_argument("--tile-rows", type=int, default=0,
        help="zero selects 64 for dynamic-leaf search and 1024 otherwise")
    parser.add_argument("--static-tile-rows", type=int, default=1024)
    parser.add_argument("--patch-instructions-per-ast", type=int, default=64)
    parser.add_argument("--final-cpu-optimize", type=int, choices=(0, 1), default=1)
    parser.add_argument("--final-cpu-optimizer-rows", type=int, default=257)
    parser.add_argument("--final-cpu-optimizer-iterations", type=int, default=8)
    parser.add_argument("--final-cpu-optimizer-restarts", type=int, default=2)
    parser.add_argument("--final-cpu-optimizer-f-calls-limit", type=int, default=10000)
    parser.add_argument(
        "--operator-profile",
        choices=("broad", "scientific", "trig", "algebraic"),
        default="broad",
    )
    parser.add_argument("--qd-column-buckets", type=int, default=1)
    parser.add_argument("--qd-transcendental-buckets", type=int, default=1)
    parser.add_argument("--archive-parent-probability", type=float, default=0.0)
    parser.add_argument("--constant-setting-credit-weight", type=float, default=0.0)
    parser.add_argument("--parsimony-coefficient", type=float, default=0.00005)
    parser.add_argument("--validation-mode", choices=("each", "final"), default=None,
        help="evaluate the held-out partition every generation or exactly once after search")
    parser.add_argument("--stop-metric", choices=("train", "validation"), default="train")
    parser.add_argument("--stop-r2", type=float, default=0.999999)
    parser.add_argument("--time-limit-seconds", type=float, default=0.0,
        help="per-dataset search wall-time limit; zero disables the limit")
    parser.add_argument("--output", type=Path, default=Path("scratch/suite/results.csv"))
    parser.add_argument("--resume", action="store_true", help="append missing problem/seed pairs to an existing CSV")
    parser.add_argument("--persistent-process", action="store_true",
        help="run all pending problem/seed pairs in one process and amortize CUDA process setup")
    parser.add_argument("--template-output", type=Path, default=None,
        help="write every generated skeleton CUBIN event and a deduplicated sibling summary CSV")
    parser.add_argument("--generation-output", type=Path, default=None,
        help="write one maturity-policy telemetry record per completed generation")
    parser.add_argument("--checkpoint-output", type=Path, default=None,
        help="copy incumbents in memory during search and score them on held-out data after search")
    parser.add_argument("--cubin-cache", type=Path, default=Path("scratch/cache/secant_sr_cubin.sqlite3"),
        help="persistent SQLite cache for generated skeleton CUBINs")
    parser.add_argument("--no-cubin-cache", action="store_true",
        help="compile every skeleton with NVRTC and bypass the persistent cache")
    parser.add_argument("--pmlb-root", type=Path, default=None,
        help="PMLB checkout containing official dataset LFS files")
    parser.add_argument("--srbench-root", type=Path, default=None,
        help="SRBench checkout providing the frozen suite manifests")
    parser.add_argument("--prepared-data-dir", type=Path, default=Path("scratch/srbench-prepared"))
    parser.add_argument("--scale-x", action="store_true")
    parser.add_argument("--scale-y", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=("secant", SRBENCH_V2_PROTOCOL, SRBENCH_V2_BLACKBOX_PROTOCOL),
        default="secant",
        help="select frozen benchmark data and assessment semantics",
    )
    parser.add_argument("--target-noise", type=float, default=0.0)
    parser.add_argument("--skip-symbolic-assessment", action="store_true",
        help="write numerical fit results without running the potentially unbounded SRBench v2 SymPy assessor")
    args = parser.parse_args()
    custom_problems = args.problems is not None
    if args.dataset_max_inputs < 0:
        parser.error("--dataset-max-inputs must be nonnegative")
    if args.constant_optimizer == "lm" and args.backend != "cubin-maturity":
        parser.error("--constant-optimizer lm requires --backend cubin-maturity")
    if (args.lm_batch_asts <= 0 or args.lm_settings_per_cta <= 0 or args.lm_starts_per_binding <= 0
            or args.lm_tile_rows <= 0 or args.lm_threads <= 0):
        parser.error("LM batch, setting-tile, row-tile, and thread counts must be positive")
    if args.constant_settings > 0 and args.lm_settings_per_cta > args.constant_settings:
        parser.error("--lm-settings-per-cta may not exceed --constant-settings")
    if args.constant_optimizer == "lm" and args.constant_settings % args.lm_starts_per_binding:
        parser.error("--constant-settings must be divisible by --lm-starts-per-binding")
    if args.protocol in (SRBENCH_V2_PROTOCOL, SRBENCH_V2_BLACKBOX_PROTOCOL):
        if args.pmlb_root is None or args.srbench_root is None:
            parser.error(f"--protocol {args.protocol} requires --pmlb-root and --srbench-root")
        permitted_suites = (
            ("feynman-all",)
            if args.protocol == SRBENCH_V2_PROTOCOL else
            ("srbench-blackbox-all", "srbench-blackbox-smoke")
        )
        if args.suite not in permitted_suites and not custom_problems:
            parser.error(
                f"--protocol {args.protocol} requires one of {', '.join(permitted_suites)} or --problems"
            )
        if args.scale_x or args.scale_y:
            parser.error(f"--protocol {args.protocol} controls input and target scaling")
        if args.protocol == SRBENCH_V2_PROTOCOL and args.target_noise not in SRBENCH_V2_TARGET_NOISES:
            parser.error(
                f"--protocol {SRBENCH_V2_PROTOCOL} target noise must be one of "
                f"{', '.join(str(value) for value in SRBENCH_V2_TARGET_NOISES)}"
            )
        if args.protocol == SRBENCH_V2_BLACKBOX_PROTOCOL and args.target_noise != 0.0:
            parser.error(f"--protocol {SRBENCH_V2_BLACKBOX_PROTOCOL} does not use target noise")
        if args.validation_mode == "each":
            parser.error(f"--protocol {args.protocol} forbids test evaluation during search")
        args.validation_mode = "final"
        if args.stop_metric == "validation":
            parser.error(f"--protocol {args.protocol} cannot stop on the held-out test partition")
        if args.seeds is None:
            args.seeds = SRBENCH_V2_SEEDS
    else:
        if args.validation_mode is None:
            args.validation_mode = "each"
        if args.seeds is None:
            args.seeds = (1, 2, 3)
    if args.problems is None:
        if args.suite in ("feynman-all", "srbench-blackbox-all"):
            if args.pmlb_root is None or args.srbench_root is None:
                parser.error(f"--suite {args.suite} requires --pmlb-root and --srbench-root")
            manifest_name = "groundtruth.csv" if args.suite == "feynman-all" else "blackbox_results.csv"
            manifest_path = args.srbench_root / "docs" / "csv" / manifest_name
            with manifest_path.open(newline="") as manifest:
                if args.suite == "feynman-all":
                    args.problems = tuple(sorted({
                        row["dataset"]
                        for row in csv.DictReader(manifest)
                        if row["data_group"] == "Feynman"
                    }))
                else:
                    args.problems = tuple(sorted({row["dataset"] for row in csv.DictReader(manifest)}))
            if not args.problems:
                parser.error(f"SRBench manifest contains no datasets for suite {args.suite}")
            if args.protocol in (SRBENCH_V2_PROTOCOL, SRBENCH_V2_BLACKBOX_PROTOCOL):
                manifest_payload = ("\n".join(args.problems) + "\n").encode()
                manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
                expected_count, expected_sha256 = (
                    (SRBENCH_V2_FEYNMAN_DATASET_COUNT, SRBENCH_V2_FEYNMAN_MANIFEST_SHA256)
                    if args.protocol == SRBENCH_V2_PROTOCOL else
                    (SRBENCH_V2_BLACKBOX_DATASET_COUNT, SRBENCH_V2_BLACKBOX_MANIFEST_SHA256)
                )
                if len(args.problems) != expected_count or manifest_sha256 != expected_sha256:
                    parser.error(
                        f"supplied SRBench manifest does not match frozen v2.0 protocol {args.protocol}: "
                        f"count={len(args.problems)} sha256={manifest_sha256}"
                    )
        else:
            args.problems = PROBLEM_SUITES[args.suite]
    if args.suite.startswith("feynman-") and args.pmlb_root is None:
        parser.error(f"--suite {args.suite} requires --pmlb-root; generated substitutes are not accepted")
    if args.suite.startswith("srbench-") and args.pmlb_root is None:
        parser.error(f"--suite {args.suite} requires --pmlb-root")
    if args.dataset_max_inputs != 0:
        if args.protocol != SRBENCH_V2_BLACKBOX_PROTOCOL:
            parser.error("--dataset-max-inputs is currently supported only by the SRBench black-box protocol")
        retained_problems = []
        for problem in args.problems:
            source = args.pmlb_root / "datasets" / problem / f"{problem}.tsv.gz"
            if not source.is_file():
                parser.error(f"missing PMLB dataset: {source}")
            with gzip.open(source, "rt", encoding="utf-8") as dataset:
                columns = dataset.readline().rstrip("\r\n").split("\t")
            if "target" not in columns:
                parser.error(f"PMLB dataset has no target column: {source}")
            num_inputs = len(columns) - 1
            if num_inputs > args.dataset_max_inputs:
                print(
                    f"problem={problem} status=skip "
                    f"reason=num_inputs_{num_inputs}_exceed_{args.dataset_max_inputs}"
                )
            else:
                retained_problems.append(problem)
        args.problems = tuple(retained_problems)
        if not args.problems:
            parser.error("--dataset-max-inputs excluded every requested problem")
    args.dynamic_leaf_fallback_problems = frozenset()
    if args.protocol == SRBENCH_V2_BLACKBOX_PROTOCOL and args.backend == "cubin-maturity":
        fallback_problems = []
        fallback_reasons = {}
        for problem in args.problems:
            source = args.pmlb_root / "datasets" / problem / f"{problem}.tsv.gz"
            if not source.is_file():
                parser.error(f"missing PMLB dataset: {source}")
            with gzip.open(source, "rt", encoding="utf-8") as dataset:
                columns = dataset.readline().rstrip("\r\n").split("\t")
            if "target" not in columns:
                parser.error(f"PMLB dataset has no target column: {source}")
            num_inputs = len(columns) - 1
            fallback_reason = dynamic_leaf_fallback_reason(
                num_inputs,
                args.dynamic_leaves,
                args.asts_per_kernel,
            )
            if fallback_reason is not None:
                fallback_problems.append(problem)
                fallback_reasons[problem] = fallback_reason
        args.dynamic_leaf_fallback_problems = frozenset(fallback_problems)
        for problem in fallback_problems:
            print(
                f"problem={problem} backend_route=cubin-dynamic-leaf "
                f"reason={fallback_reasons[problem]}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backend.startswith("cubin") and not args.no_cubin_cache:
        args.cubin_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.template_output is not None:
        args.template_output.parent.mkdir(parents=True, exist_ok=True)
    if args.generation_output is not None:
        args.generation_output.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_output is not None:
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "problem", "seed", "backend", "generation", "train_r2", "validation_r2", "complexity", "nodes",
        "depth", "unique_columns", "column_occurrences", "constants", "unary_ops", "binary_ops", "ternary_ops",
        "transcendental_ops", "constant_setting_robustness", "dynamic_promoted", "constant_promoted",
        "materialize_validation_r2", "materialize_validation_diverged",
        "archive_cells", "archive_occupied_cells", "archive_elites",
        "final_cpu_optimizer_enabled", "final_cpu_optimizer_constants", "final_cpu_optimizer_improved",
        "final_cpu_optimizer_f_calls", "final_cpu_optimizer_seconds", "pre_final_train_r2",
        "pre_final_validation_r2", "validation_mode", "stop_metric", "stop_r2", "stop_value", "stop_reached",
        "time_limit_seconds", "time_limit_reached", "search_elapsed_seconds", "termination_reason",
        "first_hit_seconds", "solved",
        "accuracy_solution", "symbolic_solution", "symbolic_error_is_zero", "symbolic_error_is_constant",
        "symbolic_fraction_is_constant", "elapsed_seconds", "best_expression", "simplified_expression",
        "symbolic_error", "symbolic_fraction", "symbolic_assessment_error",
        "estimated_uncached_elapsed_seconds", "template_cache_enabled", "template_cache_hits",
        "template_cache_misses", "template_cache_invalidations", "template_cache_open_seconds",
        "template_cache_lookup_seconds", "template_cache_store_seconds", "template_actual_compile_seconds",
        "template_estimated_uncached_compile_seconds", "template_estimated_saved_compile_seconds",
        "template_prepare_seconds",
        "mse_reducer_cache_enabled", "mse_reducer_cache_hits", "mse_reducer_cache_misses",
        "mse_reducer_cache_invalidations", "mse_reducer_cache_lookup_seconds",
        "mse_reducer_cache_store_seconds", "mse_reducer_actual_compile_seconds",
        "mse_reducer_estimated_uncached_compile_seconds", "mse_reducer_estimated_saved_compile_seconds",
        "dataset_source", "protocol", "target_noise", "source_sha256", "train_rows", "validation_rows",
        "num_inputs",
        "scale_x", "scale_y", "operator_profile", "qd_column_buckets", "qd_transcendental_buckets",
        "archive_parent_probability",
        "dynamic_max_nodes",
        "dynamic_leaves",
        "leaf_settings",
        "leaf_setting_policy",
        "mixed_dynamic_leaves",
        "mixed_refine_probability",
        "constant_settings",
        "constant_optimize_probability",
        "constant_optimize_budget",
        "constant_optimize_interval",
        "constant_optimize_random_fraction",
        "constant_optimizer",
        "constant_optimizer_iterations",
        "constant_optimizer_scale",
        "constant_optimizer_decay",
        "constant_sweep_phase",
        "lm_batch_asts",
        "lm_settings_per_cta",
        "lm_bindings",
        "lm_starts_per_binding",
        "lm_tile_rows",
        "lm_threads",
        "lm_patch_instructions",
        "lm_initial_damping",
        "constant_setting_credit_weight",
        "parsimony_coefficient",
    )
    completed: set[tuple[str, int]] = set()
    write_header = True
    output_mode = "w"
    if args.resume and args.output.exists() and args.output.stat().st_size != 0:
        with args.output.open(newline="") as existing_output:
            existing_reader = csv.DictReader(existing_output)
            if tuple(existing_reader.fieldnames or ()) != fields:
                parser.error(f"--resume output schema does not match current fields: {args.output}")
            completed = {(row["problem"], int(row["seed"])) for row in existing_reader}
        output_mode = "a"
        write_header = False
    template_records: list[dict[str, str | int]] = []
    template_output = args.template_output.open("w", newline="") if args.template_output is not None else None
    generation_output_mode = "w"
    generation_output_write_header = True
    if args.generation_output is not None and args.resume and args.generation_output.exists() and (
        args.generation_output.stat().st_size != 0
    ):
        with args.generation_output.open(newline="") as existing_generation_output:
            existing_generation_reader = csv.DictReader(existing_generation_output)
            if tuple(existing_generation_reader.fieldnames or ()) != MATURITY_STAGE_FIELDS:
                parser.error(f"--resume generation output schema does not match current fields: {args.generation_output}")
        generation_output_mode = "a"
        generation_output_write_header = False
    generation_output = (
        args.generation_output.open(generation_output_mode, newline="") if args.generation_output is not None else None
    )
    checkpoint_output_mode = "w"
    checkpoint_output_write_header = True
    if args.checkpoint_output is not None and args.resume and args.checkpoint_output.exists() and (
        args.checkpoint_output.stat().st_size != 0
    ):
        with args.checkpoint_output.open(newline="") as existing_checkpoint_output:
            existing_checkpoint_reader = csv.DictReader(existing_checkpoint_output)
            if tuple(existing_checkpoint_reader.fieldnames or ()) != CHECKPOINT_FIELDS:
                parser.error(f"--resume checkpoint output schema does not match current fields: {args.checkpoint_output}")
        checkpoint_output_mode = "a"
        checkpoint_output_write_header = False
    checkpoint_output = (
        args.checkpoint_output.open(checkpoint_output_mode, newline="")
        if args.checkpoint_output is not None else None
    )
    try:
        template_writer = (
            csv.DictWriter(template_output, fieldnames=TEMPLATE_EVENT_FIELDS, lineterminator="\n")
            if template_output is not None else None
        )
        if template_writer is not None:
            template_writer.writeheader()
        generation_writer = (
            csv.DictWriter(generation_output, fieldnames=MATURITY_STAGE_FIELDS, lineterminator="\n")
            if generation_output is not None else None
        )
        if generation_writer is not None and generation_output_write_header:
            generation_writer.writeheader()
        checkpoint_writer = (
            csv.DictWriter(checkpoint_output, fieldnames=CHECKPOINT_FIELDS, lineterminator="\n")
            if checkpoint_output is not None else None
        )
        if checkpoint_writer is not None and checkpoint_output_write_header:
            checkpoint_writer.writeheader()
        with args.output.open(output_mode, newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            if write_header:
                writer.writeheader()
            pending_jobs = []
            for problem in args.problems:
                for seed in args.seeds:
                    if (problem, seed) in completed:
                        print(f"problem={problem} seed={seed} status=resume_skip")
                    else:
                        pending_jobs.append((problem, seed))
            for (problem, seed), (record, templates, maturity_stages, checkpoints) in run_pending(
                args,
                pending_jobs,
            ):
                writer.writerow(record)
                output.flush()
                template_records.extend(templates)
                if template_writer is not None:
                    template_writer.writerows(templates)
                    template_output.flush()
                if generation_writer is not None:
                    generation_writer.writerows(maturity_stages)
                    generation_output.flush()
                if checkpoint_writer is not None:
                    checkpoint_writer.writerows(checkpoints)
                    checkpoint_output.flush()
                print(
                    f"problem={problem} seed={seed} validation_r2={record['validation_r2']:.9g} "
                    f"solved={record['solved']} elapsed_seconds={record['elapsed_seconds']:.3f} "
                    f"templates_generated={len(templates)} maturity_generations={len(maturity_stages)} "
                    f"checkpoints={len(checkpoints)}"
                )
    finally:
        if template_output is not None:
            template_output.close()
        if generation_output is not None:
            generation_output.close()
        if checkpoint_output is not None:
            checkpoint_output.close()
    if args.template_output is not None:
        summary_path = args.template_output.with_name(f"{args.template_output.stem}_summary.csv")
        template_summary_write(summary_path, template_records)
        print(
            f"template_events={len(template_records)} unique_templates="
            f"{len({str(record['template_key']) for record in template_records})} summary={summary_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
