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
"""Run one C99 secant-sr campaign and emit progress as JSON lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=Path("build/secant_sr_search"))
    parser.add_argument(
        "--backend",
        choices=("cpu", "cubin", "cubin-dynamic-leaf", "cubin-staged", "cubin-maturity"),
        default="cpu",
    )
    parser.add_argument("--problem", default="nguyen1")
    parser.add_argument("--population", type=int, default=8192)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--rows", type=int, default=257)
    parser.add_argument("--validation-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
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
    parser.add_argument("--constant-optimizer-decay", type=float, default=0.5)
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
    parser.add_argument("--qd-column-buckets", type=int, default=1)
    parser.add_argument("--qd-transcendental-buckets", type=int, default=1)
    parser.add_argument("--archive-parent-probability", type=float, default=0.0)
    parser.add_argument("--constant-setting-credit-weight", type=float, default=0.0)
    parser.add_argument("--parsimony-coefficient", type=float, default=0.00005)
    args = parser.parse_args()
    if args.lm_starts_per_binding <= 0:
        parser.error("--lm-starts-per-binding must be positive")
    if args.constant_optimizer == "lm" and args.constant_settings % args.lm_starts_per_binding:
        parser.error("--constant-settings must be divisible by --lm-starts-per-binding")
    command = [
        str(args.executable),
        "--backend", args.backend,
        "--problem", args.problem,
        "--population", str(args.population),
        "--generations", str(args.generations),
        "--rows", str(args.rows),
        "--validation-rows", str(args.validation_rows),
        "--seed", str(args.seed),
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
        "--qd-column-buckets", str(args.qd_column_buckets),
        "--qd-transcendental-buckets", str(args.qd_transcendental_buckets),
        "--archive-parent-probability", str(args.archive_parent_probability),
        "--constant-setting-credit-weight", str(args.constant_setting_credit_weight),
        "--parsimony-coefficient", str(args.parsimony_coefficient),
    ]
    if args.constant_optimize_probability is not None:
        command.extend(("--constant-optimize-probability", str(args.constant_optimize_probability)))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line.startswith(f"problem={args.problem} generation="):
            record = dict(field.split("=", 1) for field in line.split())
            print(json.dumps(record, separators=(",", ":")))
        elif line.startswith("final_cpu_optimizer "):
            record = dict(field.split("=", 1) for field in line.split())
            print(json.dumps(record, separators=(",", ":")))
        elif " best_expression=" in line:
            print(json.dumps({"best_expression": line.split(" best_expression=", 1)[1]}, separators=(",", ":")))
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
