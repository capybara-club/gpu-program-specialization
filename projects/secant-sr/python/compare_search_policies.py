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
"""Matched-data/time policy trial; settings stays in an isolated archived binary.

Three arms: historical maturity/settings without LM, toggle GP before selector
CSE, and the identical toggle GP with CSE. Only the latter two isolate a compiler
change. Uses prepared SRBench files and the existing toggle validation adapter.
"""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

from run_srbench_toggle import DEFAULTS, atomic_json, execute, sha256


def fields(line):
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def settings_result(text, problem):
    generations = [fields(s) for s in text.splitlines()
                   if s.startswith(f"problem={problem} generation=")]
    audits = [fields(s) for s in text.splitlines() if s.startswith("final_cpu_optimizer ")]
    if not generations or not audits:
        raise ValueError("missing generation or final CPU audit")
    if any(int(g.get("materialize_validation_diverged", "0")) for g in generations):
        raise ValueError("legacy CPU/GPU materialization diverged")
    audit = audits[-1]
    if int(audit["enabled"]) != 0:
        raise ValueError("unexpected final CPU optimization")
    train, test = float(audit["train_r2_after"]), float(audit["validation_r2_after"])
    if not math.isfinite(train) or not math.isfinite(test):
        raise ValueError("nonfinite legacy winner")
    g = generations[-1]
    expressions = [s.split(" best_expression=", 1)[1] for s in text.splitlines()
                   if s.startswith(f"problem={problem} best_expression=")]
    if not expressions:
        raise ValueError("missing legacy winner expression")
    return dict(status="completed", train_r2=train, validation_r2=test,
                accuracy_solution=int(test > .999), best_expression=expressions[-1],
                elapsed_seconds=float(g["elapsed_seconds"]), generations=int(g["generation"])+1,
                stop_reason="training_threshold" if int(g["stop_reached"]) else
                "time_budget" if int(g["time_limit_reached"]) else "generation_budget",
                symbolic_status="not_assessed")


def legacy(job, data_dir, executable, output, cache, gpu, seconds, warmup=False):
    output.mkdir(parents=True, exist_ok=False)
    data = data_dir / job["path"]
    record = dict(problem=job["problem"], seed=job["seed"], status="failed")
    command = [str(executable), "--backend", "cubin-maturity", "--problem", job["problem"],
               "--dataset-binary", str(data), "--seed", str(job["seed"]),
               "--population", "8192", "--generations", "1" if warmup else "1000",
               "--time-limit-seconds", str(seconds), "--leaf-settings", "4096",
               "--constant-settings", "512", "--constant-optimizer", "legacy",
               "--constant-optimize-budget", "512", "--constant-optimize-interval", "1",
               "--constant-optimizer-iterations", "4", "--final-cpu-optimize", "0",
               "--operator-profile", "scientific", "--validation-mode", "final",
               "--stop-metric", "train", "--stop-r2", "0.999999",
               "--workers", "3", "--streams", "8",
               "--cubin-cache", str(cache)]
    record["command"] = command
    try:
        if sha256(data) != job["prepared_sha256"]:
            raise ValueError("prepared dataset changed")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MODULE_LOADING="EAGER")
        with (output/"stdout.log").open("w") as out, (output/"stderr.log").open("w") as err:
            begin = time.perf_counter()
            process = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
            try:
                process.wait(timeout=max(180, 2*seconds+120))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record["status"] = "timeout"
            record["process_wall_seconds"] = time.perf_counter()-begin
        if process.returncode or record["status"] == "timeout":
            raise RuntimeError(f"legacy exit {process.returncode}; see retained logs")
        record.update(settings_result((output/"stdout.log").read_text(), job["problem"]))
    except Exception as error:
        record["error"] = str(error)
    atomic_json(output/"result.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", type=Path, nargs="+", required=True)
    for name in ("settings", "before", "after", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--gpu", default="1")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--datasets", type=int, default=12)
    p.add_argument("--selection-seed", type=int, default=20260919)
    a = p.parse_args()
    if a.seconds <= 0 or not math.isfinite(a.seconds):
        p.error("seconds must be finite and positive")
    manifests = [(path.resolve(), json.loads(path.read_text())) for path in a.manifests]
    names = sorted(set(j["problem"] for j in manifests[0][1]["jobs"]))
    if not 1 <= a.datasets <= len(names):
        p.error("dataset count outside manifest")
    chosen = random.Random(a.selection_seed).sample(names, a.datasets)
    jobs = []
    for path, manifest in manifests:
        available = collections.defaultdict(list)
        for job in manifest["jobs"]:
            available[job["problem"]].append(job)
        if any(n not in available for n in chosen):
            raise ValueError("manifests cover different datasets")
        for name in chosen:
            jobs.extend((path.parent, j) for j in available[name])
    if len({j["id"] for _, j in jobs}) != len(jobs):
        raise ValueError("duplicate dataset/seed jobs")
    binaries = {name: getattr(a, name).resolve() for name in ("settings", "before", "after")}
    config = dict(DEFAULTS, generations=1000, seconds=a.seconds)
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    plan = dict(chosen=chosen, selection_seed=a.selection_seed, gpu=a.gpu, seconds=a.seconds,
                cases=len(jobs), arms=list(binaries), toggle_config=config,
                started_unix=time.time(), binary_sha256={n:sha256(b) for n,b in binaries.items()},
                manifest_sha256={str(p):sha256(p) for p,_ in manifests},
                runner_sha256=sha256(Path(__file__)),
                adapter_sha256={n:sha256(Path(__file__).with_name(n)) for n in
                                ("run_srbench_toggle.py", "secant_sr_ast.py")},
                jobs=[j for _,j in jobs],
                limitations=["Different GP policies: settings has random coefficient refinement; toggles has fixed banks, no LM",
                  "Different tree bounds, literal vocabularies and internal batch geometry; this is not a selector-only search ablation",
                  "Same prepared data, seed, GPU, population, operator set and nominal 30-second budget",
                  "Generation-boundary budgets may overrun; process wall time is retained",
                  "One-generation warmup per input count/arm excluded and retained separately",
                  "Only before/after toggles isolate the compiler change; GPU sums can change close ties",
                  "Numerical held-out R2 > 0.999, symbolic assessment not performed"])
    plan["limitations"][2] = plan["limitations"][2].replace("30-second", f"{a.seconds:g}-second")
    plan["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                                               "--format=csv,noheader"], text=True)
    atomic_json(a.output/"plan.json", plan)
    records, warmed = [], set()
    for index, (data_dir, job) in enumerate(jobs):
        order = ["settings", "before", "after"]
        shift = index % 3
        order = order[shift:]+order[:shift]
        for arm in order:
            key = (arm, job["num_inputs"])
            if key not in warmed:
                dest = a.output/"warmups"/f"{arm}-inputs{job['num_inputs']}"
                if arm == "settings":
                    warm = legacy(job, data_dir, binaries[arm], dest,
                                  a.output/"settings-templates.sqlite", a.gpu, a.seconds, True)
                else:
                    warm = execute(job, data_dir, binaries[arm], dict(config, generations=1),
                                   "cuda", a.gpu, dest, max(180,2*a.seconds+120))
                if warm["status"] != "completed":
                    atomic_json(a.output/"failure.json", dict(stage="warmup", arm=arm, record=warm))
                    raise RuntimeError(f"{arm} warmup failed; campaign paused")
                warmed.add(key)
            dest = a.output/arm
            if arm == "settings":
                result = legacy(job, data_dir, binaries[arm], dest/"trials"/job["id"],
                                a.output/"settings-templates.sqlite", a.gpu, a.seconds)
            else:
                result = execute(job, data_dir, binaries[arm], config, "cuda", a.gpu, dest, max(180,2*a.seconds+120))
            result["arm"] = arm
            records.append(result)
            atomic_json(a.output/"results.json", records)
            summary = {}
            for name in binaries:
                rows = [r for r in records if r["arm"] == name]
                summary[name] = dict(finished=len(rows), planned=len(jobs),
                    errors=sum(r["status"] != "completed" for r in rows),
                    successes=sum(r.get("accuracy_solution",0) for r in rows),
                    process_seconds=sum(r.get("process_wall_seconds",0) for r in rows))
            atomic_json(a.output/"progress.json", summary)
            print(f"{index+1}/{len(jobs)} {arm} {job['id']} {result['status']} "
                  f"R2={result.get('validation_r2')} wall={result.get('process_wall_seconds')}", flush=True)
            if result["status"] != "completed":
                raise RuntimeError("retained failed trial; campaign paused instead of excluding it")
    plan["finished_unix"] = time.time()
    atomic_json(a.output/"plan.json", plan)


if __name__ == "__main__":
    main()
