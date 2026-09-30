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
"""SRBench v2 data preparation and resumable Secant 0.3 toggle-GP campaigns.

Only `prepare` needs NumPy/pandas/scikit-learn. Execution/reporting use the
standard library. Truth formulas never enter the search process.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import signal
import statistics
import struct
import subprocess
import threading
import time

from srbench_v2 import (
    SRBENCH_V2_BLACKBOX_DATASET_COUNT, SRBENCH_V2_BLACKBOX_MANIFEST_SHA256,
    SRBENCH_V2_BLACKBOX_PROTOCOL, SRBENCH_V2_FEYNMAN_DATASET_COUNT,
    SRBENCH_V2_FEYNMAN_MANIFEST_SHA256, SRBENCH_V2_PROTOCOL,
    SRBENCH_V2_SEEDS, SRBENCH_V2_TARGET_NOISES,
)

SCHEMA = 1
DEFAULTS = {
    "population": 8192, "generations": 100, "seconds": 30,
    "banks": 64, "constants": 4, "toggle_bits": 6,
    "distribution": "uniform", "constant_min": -2, "constant_max": 2,
    "max_nodes": 31, "max_depth": 10, "initial_depth": 3, "elites": 8,
    "operators": "add,sub,mul,div,sin,cos,sqrt,exp,log",
    "stop_nmse": 1e-6, "parsimony": 1e-6,
    "toggle_probability": 0.8, "four_way_probability": 0.3,
    "coefficient_probability": 0.25, "ast_batch": 4096, "score_mib": 256,
    "pack": 8, "kernels": 16, "tile_rows": 128, "threads": 128,
    "workers": 3, "streams": 8,
}
GPU_OPTIONS = {"pack", "kernels", "tile_rows", "threads", "workers", "streams"}
FIELDS = (
    "problem", "seed", "protocol", "target_noise", "status", "stop_reason",
    "train_r2", "validation_r2", "accuracy_solution", "symbolic_solution",
    "train_mse", "validation_mse", "ast_occurrences", "configurations",
    "elapsed_seconds", "process_wall_seconds", "scoring_seconds", "setup_seconds",
    "nvrtc_seconds", "device_seconds", "module_load_seconds", "reduction_seconds",
    "transfer_seconds", "train_rows", "validation_rows", "num_inputs", "gpu",
    "refinement_configurations", "total_configurations", "refinement_seconds",
    "refinement_models", "refinement_skipped_parameter_capacity",
    "best_expression", "source_sha256", "prepared_sha256", "error",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest_names(root, suite):
    filename = "groundtruth.csv" if suite == "feynman" else "blackbox_results.csv"
    with (root / "docs/csv" / filename).open(newline="") as stream:
        names = sorted({row["dataset"] for row in csv.DictReader(stream)
                        if suite != "feynman" or row["data_group"] == "Feynman"})
    expected = (SRBENCH_V2_FEYNMAN_DATASET_COUNT, SRBENCH_V2_FEYNMAN_MANIFEST_SHA256)
    if suite == "blackbox":
        expected = (SRBENCH_V2_BLACKBOX_DATASET_COUNT, SRBENCH_V2_BLACKBOX_MANIFEST_SHA256)
    digest = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    if (len(names), digest) != expected:
        raise ValueError(f"frozen {suite} manifest mismatch: {len(names)} names, {digest}")
    return names, digest


def prepare(args):
    import numpy as np
    import pandas as pd
    import sklearn
    from srbench_data import (
        HEADER, MAGIC, VERSION, prepare_srbench_v2_blackbox_dataset,
        prepare_srbench_v2_groundtruth_dataset,
    )

    names, manifest_hash = manifest_names(args.srbench_root, args.suite)
    selected = names if not args.problems else sorted(set(args.problems))
    if set(selected) - set(names):
        raise ValueError("requested problems are outside the frozen suite")
    if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) - set(SRBENCH_V2_SEEDS):
        raise ValueError("choose distinct official SRBench v2 seeds")
    if args.noise not in SRBENCH_V2_TARGET_NOISES or (args.suite == "blackbox" and args.noise != 0):
        raise ValueError("unsupported target-noise level for this protocol")
    sources = {n: args.pmlb_root / "datasets" / n / f"{n}.tsv.gz" for n in selected}
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} dataset payloads missing; first: {missing[0]}")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "manifest.json"
    if destination.exists():
        raise ValueError("prepared manifest already exists; use a new output directory")
    jobs = []
    for name in selected:
        for seed in args.seeds:
            if args.suite == "feynman":
                item = prepare_srbench_v2_groundtruth_dataset(sources[name], args.output / "data", seed, args.noise)
            else:
                item = prepare_srbench_v2_blackbox_dataset(sources[name], args.output / "data", seed)
            raw = item.path.read_bytes()
            magic, version, inputs, train, test = HEADER.unpack_from(raw)
            if (magic, version, inputs, train, test) != (
                MAGIC, VERSION, item.num_inputs, item.num_train_rows, item.num_validation_rows
            ) or len(raw) != HEADER.size + 4 * (inputs + 1) * (train + test):
                raise ValueError(f"invalid cached prepared data: {item.path}")
            values = np.frombuffer(raw, dtype="<f4", offset=HEADER.size)
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite prepared data: {item.path}")
            train_y = values[inputs * train:(inputs + 1) * train].astype(np.float64)
            test_y = values[-test:].astype(np.float64)
            metadata = asdict(item)
            metadata["path"] = str(item.path.relative_to(args.output))
            metadata["source_path"] = str(item.source_path.resolve())
            metadata.update(problem=name, seed=seed, id=f"{name}-s{seed}",
                            prepared_sha256=hashlib.sha256(raw).hexdigest(),
                            train_variance=float(np.var(train_y)), test_variance=float(np.var(test_y)))
            jobs.append(metadata)
        print(f"prepared {name} ({len(jobs)} trials)", flush=True)
    atomic_json(destination, {
        "schema": SCHEMA, "suite": args.suite, "official_dataset_count": len(names),
        "official_manifest_sha256": manifest_hash, "problems": selected,
        "seeds": args.seeds, "noise": args.noise, "jobs": jobs,
        "data_precision": "float32", "assessment_precision": "native_float32_sse",
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    })
    print(f"manifest={destination} trials={len(jobs)}", flush=True)


def load_manifest(path):
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or value.get("suite") not in {"feynman", "blackbox"} or not value.get("jobs"):
        raise ValueError("unsupported or empty manifest")
    seen = set()
    for job in value["jobs"]:
        if job["id"] in seen or Path(job["id"]).name != job["id"] or job["id"] in {".", ".."}:
            raise ValueError("duplicate or unsafe job id")
        seen.add(job["id"])
        path_parts = Path(job["path"])
        if path_parts.is_absolute() or ".." in path_parts.parts:
            raise ValueError("prepared dataset must be inside manifest directory")
    return value


def configuration(path):
    config = dict(DEFAULTS)
    if path:
        given = json.loads(path.read_text())
        optional = {"bank_seed", "mean", "stddev", "refine_rounds", "refine_budget", "refine_scale",
                    "align_crossover_bits", "toggle_mutation_probability", "leaf_mix_probability",
                    "refine_parameters", "power_mutation_probability", "finalists",
                    "lm_iterations", "lm_budget", "lm_bindings", "lm_starts", "lm_parameters",
                    "lm_interval", "lm_threads", "lm_scale"}
        if not isinstance(given, dict) or set(given) - (set(DEFAULTS) | optional):
            raise ValueError("unsupported search configuration keys")
        config.update(given)
    if not isinstance(config["seconds"], (int, float)) or not math.isfinite(config["seconds"]) or config["seconds"] <= 0:
        raise ValueError("seconds must be finite and positive")
    if any(not isinstance(v, (int, float, str)) or isinstance(v, bool) for v in config.values()):
        raise ValueError("configuration values must be scalar strings/numbers")
    return config


def command_for(executable, data, job, config, backend):
    command = [str(executable), "--backend", backend, "--data", str(data), "--seed", str(job["seed"])]
    for key, value in sorted(config.items()):
        if backend == "cpu" and key in GPU_OPTIONS:
            continue
        command += ["--" + key.replace("_", "-"), str(value)]
    return command


def r2(mse, variance):
    if mse is None or not math.isfinite(mse):
        return None
    return 1 - mse / variance if variance > 0 else float(mse == 0)


def execute(job, manifest_dir, executable, config, backend, gpu, output, timeout):
    record = {"id": job["id"], "problem": job["problem"], "seed": job["seed"],
              "protocol": job["protocol"], "target_noise": job["target_noise"],
              "status": "failed", "gpu": gpu, "symbolic_solution": "",
              "symbolic_status": "not_assessed", "train_rows": job["num_train_rows"],
              "validation_rows": job["num_validation_rows"], "num_inputs": job["num_inputs"],
              "source_sha256": job["source_sha256"], "prepared_sha256": job["prepared_sha256"]}
    folder = output / "trials" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (folder / f"attempt-{attempt}.jsonl").exists():
        attempt += 1
    stdout_path = folder / f"attempt-{attempt}.jsonl"
    stderr_path = folder / f"attempt-{attempt}.stderr"
    record["attempt"] = attempt
    started = time.perf_counter()
    try:
        data = manifest_dir / job["path"]
        if sha256(data) != job["prepared_sha256"]:
            raise ValueError("prepared dataset hash changed")
        with data.open("rb") as stream:
            header = struct.unpack("<8sIIQQ", stream.read(32))
        expected = (b"SECSRDS\0", 1, job["num_inputs"], job["num_train_rows"], job["num_validation_rows"])
        if header != expected or data.stat().st_size != 32 + 4 * (header[2] + 1) * (header[3] + header[4]):
            raise ValueError("prepared data dimensions differ from manifest")
        if job["num_inputs"] + config["constants"] > 128:
            raise ValueError("input columns plus constant slots exceed 128; no fallback/reduction applied")
        command = command_for(executable, data, job, config, backend)
        record["command"] = command
        env = dict(os.environ, CUDA_MODULE_LOADING="EAGER")
        if backend == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            begin = time.perf_counter()
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record["status"] = "timeout"
            record["process_wall_seconds"] = time.perf_counter() - begin
            record["exit_code"] = process.returncode
        records = [json.loads(line) for line in stdout_path.read_text().splitlines() if line.strip()]
        if record["status"] == "timeout":
            raise TimeoutError(f"process exceeded hard deadline {timeout}s; partial scores not accepted")
        if process.returncode:
            raise RuntimeError(f"exit {process.returncode}: {stderr_path.read_text()[-4000:]}")
        if not records or records[-1].get("event") != "result":
            raise ValueError("missing final result")
        result = records[-1]
        record["native_result"] = result
        record["finalists"] = next((v for v in records if v.get("event")=="finalists"), None)
        record["stop_reason"] = result["status"]
        progress = next((r for r in reversed(records) if r.get("event") == "generation"), {})
        record["ast_occurrences"] = result.get("ast_occurrences", progress.get("ast_occurrences", 0))
        record["configurations"] = result.get("configurations", progress.get("configurations", "0"))
        refinement = result.get("refinement", {})
        record["refinement_configurations"] = str(refinement.get("configurations", "0"))
        record["total_configurations"] = str(int(record["configurations"]) + int(record["refinement_configurations"]))
        record["refinement_seconds"] = refinement.get("seconds", 0)
        record["lm"] = result.get("lm", {})
        record["refinement_models"] = refinement.get("models", 0)
        record["refinement_skipped_parameter_capacity"] = refinement.get("skipped_parameter_capacity", 0)
        record["refinement_skipped_inactive"] = refinement.get("skipped_inactive", 0)
        record["power_mutations"] = result.get("variation", {}).get("power_mutations", 0)
        record["elapsed_seconds"] = result.get("timing", {}).get("total_seconds", record["process_wall_seconds"])
        record.update(result.get("timing", {}))
        if result["status"] == "no_finite_model":
            record.update(status="no_finite_model", accuracy_solution=0 if job["protocol"] == SRBENCH_V2_PROTOCOL else "")
        else:
            from secant_sr_ast import Expression
            genome = Expression.decode(bytes.fromhex(result["genotype_hex"]))
            model = Expression.decode(bytes.fromhex(result["resolved_ast_hex"]))
            if genome.resolve(result["coefficients"], result["permutation"]) != model:
                raise ValueError("winner indices/coefficients do not replay to resolved AST")
            train_r2 = r2(result["train_mse"], job["train_variance"])
            test_r2 = r2(result["validation_mse"], job["test_variance"])
            record.update(status="completed", train_r2=train_r2, validation_r2=test_r2,
                          train_mse=result["train_mse"], validation_mse=result["validation_mse"],
                          best_expression=result["expression"],
                          accuracy_solution=int(test_r2 is not None and test_r2 > .999)
                          if job["protocol"] == SRBENCH_V2_PROTOCOL else "")
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["job_wall_seconds"] = time.perf_counter() - started
    atomic_json(folder / "result.json", record)
    return record


def summarize(jobs, records):
    completed = [r for r in records.values() if r["status"] in {"completed", "no_finite_model"}]
    errors = [r for r in records.values() if r["status"] not in {"completed", "no_finite_model"}]
    values = [r["validation_r2"] for r in completed if r.get("validation_r2") is not None]
    successes = sum(r.get("accuracy_solution") == 1 for r in completed)
    return {"planned": len(jobs), "finished": len(records), "completed": len(completed),
            "errors": len(errors), "pending": len(jobs) - len(records),
            "numerical_successes": successes, "finite_r2_count": len(values),
            "median_validation_r2": statistics.median(values) if values else None,
            "process_seconds_sum": sum(r.get("process_wall_seconds", 0) for r in records.values()),
            "ast_occurrences": sum(int(r.get("ast_occurrences", 0)) for r in records.values()),
            "configurations": str(sum(int(r.get("configurations", 0)) for r in records.values())),
            "refinement_configurations": str(sum(int(r.get("refinement_configurations", 0)) for r in records.values())),
            "total_configurations": str(sum(int(r.get("total_configurations", r.get("configurations", 0))) for r in records.values())),
            "symbolic_status": "not_assessed", "updated_unix": time.time()}


def write_reports(output, jobs, records):
    atomic_json(output / "progress.json", summarize(jobs, records))
    temporary = output / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, extrasaction="ignore")
        writer.writeheader()
        for job in jobs:
            writer.writerow(records.get(job["id"], {"problem": job["problem"], "seed": job["seed"], "status": "pending"}))
    temporary.replace(output / "results.csv")


def run(args):
    manifest = load_manifest(args.manifest)
    config = configuration(args.config)
    executable = args.executable.resolve()
    jobs = manifest["jobs"]
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        raise ValueError("GPU indices must be distinct and nonnegative")
    if args.backend == "cpu" and len(args.gpus) != 1:
        raise ValueError("CPU reference campaigns use one worker")
    deadline = args.hard_timeout or 2 * config["seconds"] + 120
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("hard timeout must be positive")
    identity = {"schema": SCHEMA, "manifest_sha256": sha256(args.manifest),
                "executable_sha256": sha256(executable), "config": config,
                "backend": args.backend, "gpus": args.gpus, "hard_timeout": deadline,
                "source_sha256": {name: sha256(Path(__file__).with_name(name))
                                  for name in (Path(__file__).name, "secant_sr_ast.py", "srbench_data.py", "srbench_v2.py")}}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = args.output / "campaign.json"
        if identity_path.exists():
            old = json.loads(identity_path.read_text())
            if not args.resume or old["identity"] != identity:
                raise ValueError("existing campaign requires --resume and identical code/data/configuration")
        else:
            if any(p.name != ".lock" for p in args.output.iterdir()):
                raise ValueError("refusing a nonempty output directory without campaign identity")
            hardware = "CPU reference"
            if args.backend == "cuda":
                try:
                    probe = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
                                           text=True, capture_output=True, timeout=10, check=True)
                    hardware = probe.stdout.strip()
                except (OSError, subprocess.SubprocessError) as error:
                    hardware = f"hardware query unavailable: {error}"
            atomic_json(identity_path, {"identity": identity, "manifest": manifest,
                                       "hostname": platform.node(), "started_unix": time.time(),
                                       "hardware": hardware,
                                       "deviations": ["new GP policy; GPU interpreter LM enabled" if config.get("lm_iterations",0) else "new GP policy; no LM", "float32 inputs and SSE",
                                                      "symbolic assessment deferred", "NVRTC initialization per fit",
                                                      "fixed base bank; optional native-toggle refinement and configured logical node cap; no exact representability claim"]})
        records = {}
        pending = queue.Queue()
        for job in jobs:
            path = args.output / "trials" / job["id"] / "result.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("id") != job["id"]:
                    raise ValueError("stored result has wrong job identity")
                if not args.retry_failed or record["status"] in {"completed", "no_finite_model"}:
                    records[job["id"]] = record
                    continue
            pending.put(job)
        guard = threading.Lock()
        write_reports(args.output, jobs, records)

        def worker(gpu):
            while True:
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                print(f"start gpu={gpu} problem={job['problem']} seed={job['seed']}", flush=True)
                record = execute(job, args.manifest.parent, executable, config, args.backend,
                                 gpu, args.output, deadline)
                with guard:
                    records[job["id"]] = record
                    write_reports(args.output, jobs, records)
                    print(f"finish {len(records)}/{len(jobs)} gpu={gpu} problem={job['problem']} "
                          f"status={record['status']} r2={record.get('validation_r2')} "
                          f"wall={record.get('process_wall_seconds')}", flush=True)

        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in args.gpus]
            for future in futures:
                future.result()
        summary = summarize(jobs, records)
        print(json.dumps(summary, allow_nan=False), flush=True)
        return int(summary["errors"] > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--suite", choices=("feynman", "blackbox"), required=True)
    p.add_argument("--pmlb-root", type=Path, required=True)
    p.add_argument("--srbench-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[SRBENCH_V2_SEEDS[0]])
    p.add_argument("--problems", nargs="+")
    p.add_argument("--noise", type=float, default=0)
    p = commands.add_parser("run")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--hard-timeout", type=float)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p = commands.add_parser("status")
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args)
            return 0
        if args.action == "status":
            print((args.output / "progress.json").read_text())
            return 0
        return run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
