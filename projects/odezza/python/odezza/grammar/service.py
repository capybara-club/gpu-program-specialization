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
"""Durable direct jobs, independent of the existing GP/search service."""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import fcntl
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

from odegrammar.compiler import Compiler, SQLiteDedup, canonical_json, content_id
from . import VERSION
from .lowering import CompatibilityError, lower_system
from .prelude import PROFILE
from .problem import prepare_problem, attach_knowns, resolve_integration, cpu_score
from .retention import retention_k, local_retention_k, make_report
from .results import ADDRESS_VERSION, compact_report, reconstruct


DEFAULT_EXECUTION = dict(batch_variants=64, module_systems=64, patch_capacity=384,
                         max_chunk_configurations=1048576, max_device_bytes=256*1024*1024,
                         max_bank_bytes=256*1024*1024, max_seconds=600.)


def execution_options(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value)-set(DEFAULT_EXECUTION):
        raise ValueError("Unknown execution option")
    result = {**DEFAULT_EXECUTION, **value}
    for name, item in result.items():
        if name == "max_seconds":
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
                raise ValueError("max_seconds must be positive and finite")
        elif type(item) is not int or item <= 0:
            raise ValueError(name+" must be a positive integer")
    if result["batch_variants"] > 4096 or result["module_systems"] > 256 or result["patch_capacity"] > 4096:
        raise CompatibilityError("Execution batch/module/patch limit exceeded (4096/256/4096)")
    if result["max_device_bytes"] > 8*1024**3 or result["max_bank_bytes"] > 2*1024**3 or result["max_chunk_configurations"] > 2**32:
        raise CompatibilityError("Execution limits: 8 GiB scratch, 2 GiB RNG banks, 2^32 configurations per chunk")
    return result


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temporary.write_text(canonical_json(value)+"\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def capabilities():
    return dict(version=VERSION, grammar="odegrammar.postorder.v1", rng_profile=PROFILE,
                solvers=["rk4"], time_leaf=False, precision="float32", toggle_arities=[2, 4],
                max_toggle_bits=32, scalar_pool_index_bits=64,
                state_constraint="2*states + coefficient_slots + 1 < 255; template resources may impose lower limits",
                integer_powers=[-16, 16], lm_max_states=8, lm_max_fitted_parameters=8,
                observations=["irregular_times", "sparse_masks", "unobserved_states_with_explicit_initial_values"],
                objective="pooled_observed_mse_excluding_initials_v1", weighted_loss=False,
                rng_transforms=["identity", "affine", "uniform", "normal", "log_uniform"],
                rng_location="separate resident GPU generator and prelude; fixed coefficients before RK4",
                retention_units=["numeric_candidate", "resolved_structure", "variant"],
                known_rhs="full vector assembled and checked before canonical grammar compilation",
                submission="one inline problem plus grammar, or a reusable problem_id; preparation runs asynchronously",
                result_formats=["full", "compact"], replay_address=ADDRESS_VERSION,
                cancel="cooperative between compiler records and native scoring launches",
                execution_defaults=DEFAULT_EXECUTION)


class Plan:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS skeletons(id TEXT PRIMARY KEY, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS variants(id TEXT PRIMARY KEY, skeleton TEXT NOT NULL, grouping TEXT NOT NULL, data TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS variant_groups ON variants(grouping);
          CREATE TABLE IF NOT EXISTS variant_ordinals(ordinal INTEGER PRIMARY KEY, variant TEXT NOT NULL UNIQUE);
          CREATE TABLE IF NOT EXISTS provenance(id TEXT PRIMARY KEY, variant TEXT NOT NULL, data TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS provenance_variant ON provenance(variant);
          CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, variant TEXT NOT NULL, permutation INTEGER NOT NULL, numeric_id TEXT NOT NULL, mse REAL NOT NULL, data TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS candidate_group ON candidates(variant,permutation,mse);
        """)

    def compile(self, request, problem, directory, cancelled=lambda: False, options=None):
        start = time.monotonic()
        source = attach_knowns(request, problem)
        options = execution_options(options)
        dedup_path = Path(directory)/"compiler-dedup.sqlite"
        if dedup_path.exists(): raise ValueError("Generation requires a fresh job directory")
        dedup = SQLiteDedup(dedup_path)
        compiler = Compiler(source, dedup=dedup)
        integration = resolve_integration(compiler.integration["settings"], problem)
        if retention_k(compiler.retention) > 256:
            raise CompatibilityError("Retention k is limited to 256 per requested group")
        last_skeleton = None
        ordinal, address_digest = 0, hashlib.sha256()
        try:
            for record in compiler.records():
                if cancelled():
                    raise InterruptedError("Cancelled during grammar generation")
                kind = record["type"]
                if kind == "manifest":
                    manifest = dict(record)
                elif kind == "skeleton":
                    last_skeleton = record
                    self.db.execute("INSERT INTO skeletons VALUES(?,?)", (record["id"], canonical_json(record)))
                elif kind == "variant":
                    skeleton = last_skeleton if last_skeleton and last_skeleton["id"] == record["skeleton_id"] else self.skeleton(record["skeleton_id"])
                    layout, programs = lower_system(skeleton, record, problem["states"])
                    banks = {r["key"]: r["count"]*4 for r in record["pools"]["rng_bank_requests"]}
                    if sum(banks.values()) > options["max_bank_bytes"]:
                        raise CompatibilityError("Variant RNG banks exceed execution.max_bank_bytes")
                    if layout.permutation_count > options["max_chunk_configurations"]:
                        raise CompatibilityError("One toggle product exceeds max_chunk_configurations; increase the explicit execution limit")
                    minimum = layout.permutation_count*(16*(32+4*len(layout.slots))+64)+4*len(layout.slots)+64
                    if minimum >= options["max_device_bytes"]:
                        raise CompatibilityError("One variant's reduction/score workspace exceeds max_device_bytes")
                    # All compatibility checks run before any GPU submission.
                    grouping = canonical_json([len(layout.slots), layout.toggle_bits, layout.numeric_count])
                    self.db.execute("INSERT INTO variants VALUES(?,?,?,?)", (record["id"], record["skeleton_id"], grouping, canonical_json(record)))
                    if ordinal >= 2**32:
                        raise CompatibilityError("Variant ordinal exceeds uint32 address space")
                    self.db.execute("INSERT INTO variant_ordinals VALUES(?,?)", (ordinal, record["id"]))
                    address_digest.update(f"{ordinal}:{record['id']}\n".encode("ascii"))
                    ordinal += 1
                if kind in ("variant", "provenance"):
                    provenance = {k: record[k] for k in ("variant_id", "family_id", "tags", "annotations")}
                    self.db.execute("INSERT OR IGNORE INTO provenance VALUES(?,?,?)", (content_id(provenance), provenance["variant_id"], canonical_json(provenance)))
            self.db.commit()
            manifest["addressing"] = dict(version=ADDRESS_VERSION, variant_count=ordinal,
                variant_table_sha256=address_digest.hexdigest(), configuration_order="last_axis_fastest",
                coefficient_replay="retained_exact_float32_values")
            manifest["manifest_id"] = content_id(manifest)
            summary = dict(compiler.summary, adapter_compile_seconds=time.monotonic()-start)
            atomic(Path(directory)/"manifest.json", manifest)
            atomic(Path(directory)/"generation.json", summary)
            return manifest, integration, summary
        finally:
            dedup.close()

    def skeleton(self, identifier):
        row = self.db.execute("SELECT data FROM skeletons WHERE id=?", (identifier,)).fetchone()
        if row is None: raise KeyError("Unknown skeleton")
        return json.loads(row[0])

    def variant(self, identifier):
        row = self.db.execute("SELECT data FROM variants WHERE id=?", (identifier,)).fetchone()
        if row is None: raise KeyError("Unknown variant")
        return json.loads(row[0])

    def variant_index(self, identifier):
        row = self.db.execute("SELECT ordinal FROM variant_ordinals WHERE variant=?", (identifier,)).fetchone()
        if row is None: raise KeyError("Variant has no persisted ordinal")
        return row[0]

    def variant_at(self, ordinal):
        row = self.db.execute("SELECT v.data FROM variant_ordinals o JOIN variants v ON o.variant=v.id WHERE o.ordinal=?", (ordinal,)).fetchone()
        if row is None: raise KeyError("Unknown variant index")
        return json.loads(row[0])

    def candidate_at(self, variant, permutation, configuration):
        for raw, in self.db.execute("SELECT data FROM candidates WHERE variant=? AND permutation=?", (variant, permutation)):
            row = json.loads(raw)
            if row["configuration_index"] == configuration:
                return row
        raise KeyError("This configuration is not retained; no score or coefficient snapshot is available")

    def batches(self, states, size, options):
        group, batch, bank_sizes = None, [], {}
        for grouping, raw in self.db.execute("SELECT grouping,data FROM variants ORDER BY grouping,rowid"):
            variant = json.loads(raw); skeleton = self.skeleton(variant["skeleton_id"])
            layout, programs = lower_system(skeleton, variant, states)
            next_banks = dict(bank_sizes)
            for bank in variant["pools"]["rng_bank_requests"]:
                next_banks[bank["key"]] = max(next_banks.get(bank["key"], 0), bank["count"]*4)
            minimum = layout.permutation_count*(16*(32+4*len(layout.slots))+64)+4*len(layout.slots)+64
            limit = min(size, options["max_chunk_configurations"]//layout.permutation_count,
                        (options["max_device_bytes"]-1)//minimum)
            if batch and (grouping != group or len(batch) >= limit or sum(next_banks.values()) > options["max_bank_bytes"]):
                yield batch; batch, bank_sizes = [], {}
                next_banks = {bank["key"]: bank["count"]*4 for bank in variant["pools"]["rng_bank_requests"]}
            batch.append(dict(skeleton=skeleton, variant=variant, layout=layout, programs=programs))
            bank_sizes = next_banks
            group = grouping
        if batch: yield batch

    def retain(self, candidates, k):
        groups = set()
        for candidate in candidates:
            group = candidate["variant_id"], candidate["permutation"]
            groups.add(group)
            old = self.db.execute("SELECT id,mse FROM candidates WHERE variant=? AND permutation=? AND numeric_id=?", (*group, candidate["numeric_id"])).fetchone()
            if old and old[1] <= candidate["mse"]: continue
            if old: self.db.execute("DELETE FROM candidates WHERE id=?", (old[0],))
            self.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?)", (candidate["id"], *group, candidate["numeric_id"], candidate["mse"], canonical_json(candidate)))
        for group in groups:
            self.db.execute("DELETE FROM candidates WHERE variant=? AND permutation=? AND id NOT IN (SELECT id FROM candidates WHERE variant=? AND permutation=? ORDER BY mse,id LIMIT ?)", (*group, *group, k))
        self.db.commit()

    def reports(self, manifest):
        def rows():
            previous, provenance = None, []
            for raw, in self.db.execute("SELECT data FROM candidates ORDER BY variant,permutation,mse"):
                candidate = json.loads(raw)
                if candidate["variant_id"] != previous:
                    previous = candidate["variant_id"]
                    provenance = [json.loads(p) for p, in self.db.execute("SELECT data FROM provenance WHERE variant=? ORDER BY id", (previous,))]
                yield dict(candidate, provenance=provenance)
        return make_report(rows(), None, manifest["retention"], manifest["family_ids"], manifest["states"])

    def candidate(self, identifier):
        row = self.db.execute("SELECT data FROM candidates WHERE id=?", (identifier,)).fetchone()
        if row is None: raise KeyError("Candidate is not retained in this job")
        return json.loads(row[0])

    def close(self):
        self.db.close()


class Service:
    """One execution thread owns one persistent CUDA context. Stdio caller owns handles."""
    def __init__(self, root, *, library=None, device=0):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.lease = (self.root/"worker.lock").open("a")
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lease.close()
            raise RuntimeError("This storage root already belongs to a running grammar service") from None
        (self.root/"problems").mkdir(exist_ok=True)
        (self.root/"jobs").mkdir(exist_ok=True)
        self.library, self.device, self.executor = library, device, None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="odezza-grammar")
        self.jobs, self.events, self.lock = {}, {}, threading.Lock()
        # A restarted service reports interruption rather than pretending old GPU work completed.
        for status in (self.root/"jobs").glob("*/status.json"):
            value = read(status)
            if value["status"] in ("queued", "preparing", "generating", "running"):
                atomic(status, dict(value, status="interrupted", reason="service_restarted"))

    def gpu_executor(self):
        if self.executor is None:
            from .executor import Executor
            if self.library is None: raise CompatibilityError("Set --library/ODEZZA_CORE_LIBRARY to the built native shared library")
            self.executor = Executor(self.library, self.device)
        return self.executor

    def problem(self, identifier):
        if not re.fullmatch("[0-9a-f]{64}", identifier): raise ValueError("Invalid problem handle")
        return read(self.root/"problems"/(identifier+".json"))

    def job_path(self, identifier):
        if not re.fullmatch("[0-9a-f]{64}", identifier): raise ValueError("Invalid job handle")
        path = self.root/"jobs"/identifier
        if not path.is_dir(): raise KeyError("Unknown job")
        return path

    def prepare(self, spec):
        problem = prepare_problem(spec)
        atomic(self.root/"problems"/(problem["id"]+".json"), problem)
        return dict(problem_id=problem["id"], states=problem["states"],
                    trajectory_count=len(problem["offsets"])-1, point_count=len(problem["times"]),
                    observed_scalars=problem["observed_scalars"], objective=problem["objective"])

    def submit(self, problem_id=None, grammar=None, execution=None, *, problem=None, idempotency_key=None, plan_only=False):
        if (problem is None) == (problem_id is None):
            raise ValueError("Provide exactly one of problem or problem_id")
        if problem is not None and not isinstance(problem, dict): raise ValueError("problem must be an object")
        if problem_id is not None and (not isinstance(problem_id, str) or not re.fullmatch("[0-9a-f]{64}", problem_id)):
            raise ValueError("Invalid problem handle")
        if not isinstance(grammar, dict): raise ValueError("grammar must be an object")
        if idempotency_key is not None and not isinstance(idempotency_key, str): raise ValueError("idempotency_key must be a string")
        execution = execution_options(execution)
        if type(plan_only) is not bool: raise ValueError("plan_only must be a boolean")
        payload = dict(problem_id=problem_id, grammar=grammar, execution=execution, plan_only=plan_only)
        if problem is not None: payload["problem"] = problem
        # Snapshot caller-owned objects before enqueueing. Only JSON receipt,
        # shallow envelope checks and persistence precede the acknowledgement.
        payload = json.loads(canonical_json(payload))
        key = content_id(dict(payload=payload, idempotency_key=idempotency_key)) if idempotency_key is not None else content_id([payload, uuid.uuid4().hex])
        directory = self.root/"jobs"/key
        with self.lock:
            if directory.exists(): return dict(job_id=key, **read(directory/"status.json"))
            directory.mkdir()
            atomic(directory/"request.json", payload)
            atomic(directory/"status.json", dict(status="queued", completed_configurations=0))
            event = threading.Event(); self.events[key] = event
            self.jobs[key] = self.worker.submit(self._run, directory, payload, None, event)
        return dict(job_id=key, status="queued")

    def _run(self, directory, payload, problem, event):
        start = time.monotonic(); plan = None
        counts = dict(completed_configurations=0, valid=0, invalid=0, completed_batches=0)
        timing = dict(scoring_pipeline_seconds=0., template_prepare_seconds=0., rng_kernel_seconds=0.,
                      prelude_kernel_seconds=0., reducer_kernel_seconds=0., reduction_transfer_host_seconds=0.)
        native_timing_keys = tuple(timing)
        report = dict(version=VERSION, problem_id=payload.get("problem_id"), counts=counts, timing=timing)
        report["retention_complete"] = True
        try:
            if event.is_set(): raise InterruptedError("Cancelled before problem preparation")
            atomic(directory/"status.json", dict(status="preparing", **counts))
            prepare_start = time.monotonic()
            try:
                if "problem" in payload:
                    prepared = self.prepare(payload["problem"])
                    problem = self.problem(prepared["problem_id"])
                elif problem is None:
                    problem = self.problem(payload["problem_id"])
            finally:
                timing["problem_prepare_seconds"] = time.monotonic()-prepare_start
            report["problem_id"] = problem["id"]
            if event.is_set(): raise InterruptedError("Cancelled after problem preparation")
            plan = Plan(directory/"plan.sqlite")
            atomic(directory/"status.json", dict(status="generating", problem_id=problem["id"], **counts))
            manifest, integration, generation = plan.compile(payload["grammar"], problem, directory, event.is_set, payload["execution"])
            report.update(generation=generation, integration=integration, manifest_id=manifest["manifest_id"])
            if payload["plan_only"]:
                report.update(status="planned", gpu_executed=False)
            else:
                atomic(directory/"status.json", dict(status="running", phase="gpu_setup", problem_id=problem["id"], **counts))
                executor = self.gpu_executor()
                report["profile"] = executor.profile
                deadline = start+payload["execution"]["max_seconds"]
                def stop(): return event.is_set() or time.monotonic() >= deadline
                def on_chunk(rows, profile):
                    report["retention_complete"] &= profile["retention_complete"]
                    plan.retain(rows, local_retention_k(manifest["retention"]))
                    counts["completed_configurations"] += profile["configurations"]
                    counts["valid"] += profile["valid"]; counts["invalid"] += profile["invalid"]
                    for name in native_timing_keys:
                        timing[name] += profile["native"]["seconds"] if name == "scoring_pipeline_seconds" else profile[name]
                    with (directory/"chunks.jsonl").open("a") as output: output.write(canonical_json(profile)+"\n")
                    atomic(directory/"status.json", dict(status="running", phase="scoring", problem_id=problem["id"], **counts, elapsed_seconds=time.monotonic()-start))
                for batch in plan.batches(problem["states"], payload["execution"]["batch_variants"], payload["execution"]):
                    if stop(): break
                    executor.score_batch(problem, integration, batch, payload["execution"], local_retention_k(manifest["retention"]), on_chunk, stop)
                    counts["completed_batches"] += 1
                execution_complete = counts["completed_configurations"] == generation["configuration_visits"]
                status = "complete" if execution_complete and generation["complete"] and report["retention_complete"] else "partial"
                report.update(status=status, gpu_executed=counts["completed_configurations"] > 0,
                    execution_complete=execution_complete,
                    reason="cancelled" if event.is_set() else "wall_budget" if not execution_complete else "retention_budget" if not report["retention_complete"] else generation["stop_reason"],
                    winners=plan.reports(manifest))
        except InterruptedError as error:
            report.update(status="cancelled", error=str(error))
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            if plan is not None and (directory/"manifest.json").exists(): report["winners"] = plan.reports(read(directory/"manifest.json"))
        finally:
            report["gpu_executed"] = counts["completed_configurations"] > 0
            timing["total_seconds"] = time.monotonic()-start
            plan_seconds = report.get("generation", {}).get("adapter_compile_seconds", 0.)
            timing["host_and_setup_unattributed_seconds"] = max(0., timing["total_seconds"] - plan_seconds -
                sum(value for key, value in timing.items() if key != "total_seconds"))
            report["timing_semantics"] = dict(scoring_pipeline_seconds="native wall time including specialization, module loading and integration; individual stages unavailable in current public scoring report",
                                               host_and_setup_unattributed_seconds="wall-time residual, not a profiler attribution")
            atomic(directory/"report.json", report)
            atomic(directory/"status.json", dict(status=report["status"], problem_id=report["problem_id"], **counts, elapsed_seconds=timing["total_seconds"],
                                                 **({"error": report["error"]} if "error" in report else {})))
            if plan is not None: plan.close()
        return report

    def status(self, job_id):
        return dict(job_id=job_id, **read(self.job_path(job_id)/"status.json"))

    def results(self, job_id, *, format="full"):
        if format not in ("full", "compact"): raise ValueError("Result format must be full or compact")
        directory = self.job_path(job_id)
        if not (directory/"report.json").exists(): return self.status(job_id)
        report = read(directory/"report.json")
        if format == "full" or "winners" not in report: return report
        plan = Plan(directory/"plan.sqlite")
        try: return compact_report(report, read(directory/"manifest.json"), job_id, plan)
        finally: plan.close()

    def cancel(self, job_id):
        self.job_path(job_id)
        if job_id in self.events: self.events[job_id].set()
        return dict(job_id=job_id, cancellation_requested=job_id in self.events)

    def replay(self, job_id, candidate_id=None, *, address=None, cpu=False):
        if (candidate_id is None) == (address is None):
            raise ValueError("Provide exactly one of candidate_id or address")
        directory = self.job_path(job_id); plan = Plan(directory/"plan.sqlite")
        try:
            candidate = (reconstruct(plan, read(directory/"manifest.json"), address) if address is not None
                         else self._candidate(directory, plan, candidate_id))
        finally: plan.close()
        problem = self.problem(candidate["problem_id"])
        from .reference import describe
        candidate = describe(candidate, problem["states"])
        if cpu:
            details = cpu_score(problem, candidate["resolved_programs"], steps=candidate["integration"]["steps_per_observation"], breakdown=True)
            candidate["cpu_reference"] = dict(precision="float64_with_resolved_float32_coefficients", **(details if details is not None else {"mse": None}))
        return candidate

    @staticmethod
    def _candidate(directory, plan, identifier):
        if not isinstance(identifier, str) or not re.fullmatch("[0-9a-f]{64}", identifier):
            raise ValueError("Invalid candidate handle")
        try:
            return plan.candidate(identifier)
        except KeyError:
            path = directory/"fits"/(identifier+".json")
            if not path.is_file(): raise KeyError("Unknown retained or fitted candidate") from None
            return read(path)

    def fit(self, job_id, request, *, lanes=1):
        directory = self.job_path(job_id)
        if lanes not in (1, 2, 4, 8): raise ValueError("LM lanes must be 1,2,4,8")
        def work():
            from odegrammar.compiler import compile_lm_request
            spec = compile_lm_request(request)
            plan = Plan(directory/"plan.sqlite")
            try:
                results = []
                for identifier in spec["candidate_ids"]:
                    candidate = self._candidate(directory, plan, identifier)
                    result = self.gpu_executor().fit(self.problem(candidate["problem_id"]), plan.skeleton(candidate["skeleton_id"]),
                        plan.variant(candidate["variant_id"]), candidate, request, lanes=lanes)
                    from .reference import describe
                    result = describe(result, request["states"])
                    # Persist fits separately so screen top-k/provenance remain immutable.
                    path = directory/"fits"; path.mkdir(exist_ok=True)
                    atomic(path/(result["id"]+".json"), result)
                    results.append(result)
                return dict(status="complete", candidates=results)
            finally: plan.close()
        return self.worker.submit(work)

    def close(self):
        def finish():
            if self.executor is not None: self.executor.close()
        try:
            self.worker.submit(finish).result()
        finally:
            self.worker.shutdown()
            self.lease.close()
