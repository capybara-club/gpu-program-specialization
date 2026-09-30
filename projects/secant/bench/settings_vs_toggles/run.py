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
"""Paired settings/toggle benchmark, using existing binaries and one idle GPU."""
import argparse
import array
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--settings", type=Path, required=True)
p.add_argument("--toggles", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--gpu", default="1")
p.add_argument("--quick", action="store_true")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUDA_MODULE_LOADING="EAGER")
cases = [(pack, 8192, 16, trans) for trans in (0, 1) for pack in (1, 4, 8, 16, 32)]
cases += [(8, rows, banks, trans) for rows, banks in ((65536, 16), (8192, 1), (256, 16))
          for trans in (0, 1)]
if a.quick:
    cases = [(8, 8192, 16, 0)]
manifest = {"gpu": a.gpu, "cases": cases, "started_unix": time.time(),
            "comparison_tolerance": 2e-5, "row_evaluation": "one AST, one configuration, one row",
            "repeats": 7, "prepared_kernel_scheduler": "independent CUDA graph kernel nodes",
            "periodic_data_rows": 64,
            "binary_sha256": {name: hashlib.sha256(getattr(a, name).read_bytes()).hexdigest()
                              for name in ("settings", "toggles")},
            "harness_sha256": hashlib.sha256(Path(__file__).with_name("compare.c").read_bytes()).hexdigest()}
try:
    manifest["gpu_info"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"], text=True)
except (OSError, subprocess.CalledProcessError):
    manifest["gpu_info"] = "unavailable"
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
pairs, failures = [], []
for index, (pack, rows, banks, trans) in enumerate(cases):
    key = f"{'sin' if trans else 'arith'}-p{pack}-r{rows}-b{banks}"
    result = {}
    for name in (("settings", "toggles") if index % 2 == 0 else ("toggles", "settings")):
        prefix = a.output / f"{key}-{name}"
        cmd = [str(getattr(a, name).resolve()), str(pack), str(rows), str(banks), "128", "128", "128",
               str(trans), "7", str(prefix.resolve())]
        print(f"[{index + 1}/{len(cases)}] {key} {name}", flush=True)
        try:
            completed = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=180)
            prefix.with_suffix(".log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError(f"exit {completed.returncode}: {completed.stdout[-2000:]}")
            result[name] = json.loads(completed.stdout.strip().splitlines()[-1])
            prefix.with_suffix(".json").write_text(json.dumps(result[name], indent=2) + "\n")
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as error:
            failures.append({"case": key, "backend": name, "error": str(error)})
            print(f"FAILED {key} {name}: {error}", flush=True)
    if len(result) == 2:
        data = {}
        for name in result:
            data[name] = array.array("f")
            with (a.output / f"{key}-{name}.scores.f32").open("rb") as stream:
                data[name].frombytes(stream.read())
        expected = 128 * banks * 256
        if any(len(x) != expected for x in data.values()):
            raise RuntimeError(f"{key}: wrong number of scores")
        error = 0.0
        for x, y in zip(data["settings"], data["toggles"]):
            if not math.isfinite(x) or not math.isfinite(y):
                raise RuntimeError(f"{key}: nonfinite score")
            error = max(error, abs(x - y) / (1 + max(abs(x), abs(y))))
        if error > 2e-5:
            raise RuntimeError(f"{key}: cross-backend mismatch {error}")
        old, new = result["settings"], result["toggles"]
        if old.get("choice_groups", 1) != new.get("choice_groups", 1):
            raise RuntimeError("incomparable choice groups")
        pair = {"case": key, "choice_groups": old.get("choice_groups", 1), "packed": pack, "rows": rows, "banks": banks,
                "configs_per_ast": banks * 256, "transcendental": trans,
                "settings_regs": old["registers_max"], "toggle_regs": new["registers_max"],
                "settings_local": old["local_bytes_max"], "toggle_local": new["local_bytes_max"],
                "settings_row_evals_s": old["row_evals_per_second"],
                "toggle_row_evals_s": new["row_evals_per_second"],
                "kernel_speedup": new["row_evals_per_second"] / old["row_evals_per_second"],
                "settings_pipeline_s": old["pipeline_seconds_median"],
                "toggle_pipeline_s": new["pipeline_seconds_median"],
                "pipeline_speedup": old["pipeline_seconds_median"] / new["pipeline_seconds_median"],
                "cross_backend_max_scaled_error": error, "scores_compared": expected}
        pairs.append(pair)
        print(f"MATCH {expected} scores, scaled error {error:.3g}; "
              f"regs {pair['settings_regs']}->{pair['toggle_regs']}; "
              f"row-evals/s {pair['settings_row_evals_s']:.3g}->{pair['toggle_row_evals_s']:.3g}; "
              f"{pair['kernel_speedup']:.3f}x", flush=True)
        with (a.output / "pairs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pair))
            writer.writeheader()
            writer.writerows(pairs)
    (a.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
manifest["finished_unix"] = time.time()
manifest["passed_pairs"] = len(pairs)
manifest["failed_runs"] = len(failures)
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
if failures:
    raise SystemExit(1)
