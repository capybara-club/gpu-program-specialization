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
"""Queue/transport for the C99 request runtime; no compiler, database or GPU policy.

One runtime owns a selected device group; C schedules its worker threads. All job handles and prepared
problem handles are process-local; callers export reports they want to retain.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid


class ServiceError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def capabilities():
    return dict(backend="odezza.c99-runtime.v1", scoring=True, fitting=False,
                solver="fixed-subdivision-rk4-f32", known_rhs_prespecialized=True,
                customer_data_persistence=False, template_cache="C-owned SQLite; compiled kernels only",
                persistence="jobs live in memory",
                devices_per_runtime=8, automatic_tile_sizing=True, automatic_ast_pages=True,
                automatic_module_packing=True, optional_cuda_timing=True, per_skeleton_configuration_cap=True, job_release=True, combined_queued_submission=True,
                generation="C99; parsed once; double buffered; family-interleaved tiles",
                sampling_profile="odezza.c99-grammar.splitmix64-fisher-yates.v1",
                rng_profile="odezza.native.philox4x32-10.stream-v1",
                random_distributions=["uniform01", "normal01"],
                retention_units=["evaluation_row", "numeric_candidate", "resolved_structure", "variant"],
                variant_identity="family index + accepted AST index",
                replay="retained addresses with exact FP32 coefficient snapshots",
                limits=dict(retention_k=256, retained_tags=64, max_request_bytes=64*1024**2,
                            max_report_bytes=8*1024**2, report_preflight="conservative bound before scoring",
                            mcp_max_message_bytes=16*1024**2),
                unsupported=["LM", "durable job IDs", "random seek/checkpoint restore", "plan_only"])


class NativeService:
    supported_tools = {"odezza_"+x for x in
                       ("capabilities", "prepare", "submit", "status", "results", "cancel", "replay", "release")}
    capabilities = staticmethod(capabilities)

    def __init__(self, root, *, library=None, device=0, devices=None, max_jobs=32,
                 max_pending=8, max_request_bytes=256*1024**2, completed_ttl=3600):
        selected = list(devices) if devices is not None else [device]
        if not selected or len(selected)>8 or len(set(selected))!=len(selected) or any(
                isinstance(x, bool) or not isinstance(x, int) or x<0 or x>2147483647 for x in selected):
            raise ValueError("devices must contain 1..8 distinct nonnegative integers")
        if min(max_jobs, max_pending, max_request_bytes, completed_ttl)<=0:
            raise ValueError("Service limits must be positive")
        library = library or os.environ.get("ODEZZA_RUNTIME_LIBRARY") or str(
            Path(__file__).resolve().parents[3]/"build/runtime/libodezza_runtime.so")
        self.lib = C.CDLL(str(library))
        definitions = {
            "runtime_create": [C.c_uint, C.c_char_p, C.POINTER(C.c_void_p)],
            "runtime_create_devices": [C.POINTER(C.c_uint), C.c_size_t, C.c_char_p, C.POINTER(C.c_void_p)],
            "runtime_error": [C.c_void_p], "runtime_destroy": [C.c_void_p],
            "job_create": [C.c_char_p, C.c_size_t, C.POINTER(C.c_void_p)],
            "job_run": [C.c_void_p, C.c_void_p], "job_cancel": [C.c_void_p],
            "job_report": [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)],
            "job_status": [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)],
            "job_destroy": [C.c_void_p],
        }
        for name, args in definitions.items():
            function = getattr(self.lib, "odz_"+name)
            function.argtypes = args
            function.restype = (C.c_char_p if name == "runtime_error" else None if name in
                               ("runtime_destroy", "job_cancel", "job_destroy") else C.c_int)
        self.cache = str(Path(root).absolute())  # Creation and cache policy belong to C.
        self.device, self.devices, self.max_jobs = device, selected, max_jobs
        self.max_pending, self.max_request_bytes = max_pending, max_request_bytes
        self.completed_ttl = completed_ttl
        self.problem_times, self.problem_sizes, self.completed, self.worker_started = {}, {}, {}, {}
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="odezza-c99")
        self.runtime = C.c_void_p()
        self.jobs, self.handles, self.requests, self.failures, self.keys, self.problems = {}, {}, {}, {}, {}, {}
        self.lock = threading.RLock()
        self.transport = {}
        self.closed = False
        self.stopping = threading.Event()
        self.reaper = threading.Thread(target=self._reap_loop, name="odezza-expiry", daemon=True)
        self.reaper.start()

    def _reap_loop(self):
        while not self.stopping.wait(min(1., self.completed_ttl)):
            with self.lock:
                self._expire()

    def _request_bytes(self):
        # C keeps a second request copy; prepared values have a Python copy only.
        return 2*sum(map(len, self.requests.values())) + sum(self.problem_sizes.values())

    def _drop_job(self, identifier):
        self.lib.odz_job_destroy(self.handles.pop(identifier))
        for table in (self.jobs,self.requests,self.failures,self.transport,self.completed,self.worker_started):
            table.pop(identifier,None)
        self.keys = {k:v for k,v in self.keys.items() if v[1]!=identifier}

    def _expire(self):
        now = time.monotonic()
        for identifier, when in list(self.completed.items()):
            if now-when>=self.completed_ttl and self.jobs[identifier].done():
                self._drop_job(identifier)
        for identifier, when in list(self.problem_times.items()):
            if now-when>=self.completed_ttl:
                self.problems.pop(identifier,None)
                self.problem_times.pop(identifier,None)
                self.problem_sizes.pop(identifier,None)

    def release(self, job_id=None, problem_id=None):
        """Release terminal results and their idempotency keys, or a prepared input.

        Replaying an expired/released handle fails. Export a report first if needed.
        Running/queued jobs must be cancelled and reach a terminal state first.
        """
        if (job_id is None)==(problem_id is None):
            raise ValueError("Provide exactly one of job_id or problem_id")
        with self.lock:
            if job_id is not None:
                if job_id not in self.jobs:
                    raise KeyError("Unknown or released job")
                if not self.jobs[job_id].done():
                    raise ServiceError("job_active", "Job is active; cancel and wait before release")
                self._drop_job(job_id)
            else:
                del self.problems[problem_id]
                self.problem_times.pop(problem_id,None)
                self.problem_sizes.pop(problem_id,None)
        return dict(released=True, job_id=job_id, problem_id=problem_id)

    def prepare(self, problem):
        """Optional raw in-memory handle; C validates on submission."""
        encoded = json.dumps(problem, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode()) > 64*1024**2:
            raise ValueError("Problem exceeds 64 MiB")
        with self.lock:
            self._expire()
            if self._request_bytes()+len(encoded.encode())>self.max_request_bytes:
                raise ServiceError("admission_limit", "Service request-memory admission limit reached")
            if self.closed or len(self.problems) >= self.max_jobs:
                raise ServiceError("admission_limit", "Prepared-problem handle limit reached or service closed")
            identifier = uuid.uuid4().hex
            self.problems[identifier] = json.loads(encoded)
            self.problem_times[identifier] = time.monotonic()
            self.problem_sizes[identifier] = len(encoded.encode())
        return dict(problem_id=identifier, status="stored_unvalidated", lifetime="this service process")

    def submit(self, problem_id=None, grammar=None, execution=None, *, problem=None,
               idempotency_key=None, plan_only=False):
        if plan_only:
            raise ValueError("Native backend does not expose plan_only; use --backend python for legacy planning")
        if (problem is None) == (problem_id is None) or grammar is None:
            raise ValueError("Provide grammar and exactly one of problem or problem_id")
        with self.lock:
            self._expire()
            if self.closed:
                raise RuntimeError("Service is closed")
            if problem_id is not None:
                problem = self.problems[problem_id]
                self.problem_times[problem_id] = time.monotonic()
            request = dict(problem=problem, grammar=grammar, execution=execution or {})
            encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
            digest = hashlib.sha256(encoded).digest()
            if idempotency_key is not None and idempotency_key in self.keys:
                previous, identifier = self.keys[idempotency_key]
                if previous != digest:
                    raise ValueError("Idempotency key reused with a different request")
                return dict(job_id=identifier, status="submitted")
            if sum(not f.done() for f in self.jobs.values())>=self.max_pending:
                raise ServiceError("admission_limit", "Pending-job admission limit reached")
            if self._request_bytes()+2*len(encoded)>self.max_request_bytes:
                raise ServiceError("admission_limit", "Service request-memory admission limit reached")
            if len(self.handles) >= self.max_jobs:
                raise ServiceError("admission_limit", "In-memory job limit reached; release completed jobs or wait for expiration")
            handle = C.c_void_p()
            if self.lib.odz_job_create(encoded, len(encoded), C.byref(handle)):
                raise ValueError("Native job allocation failed or request exceeds 64 MiB")
            identifier = uuid.uuid4().hex
            self.handles[identifier] = handle
            self.requests[identifier] = encoded
            queued = time.monotonic()
            self.jobs[identifier] = self.executor.submit(self._run, identifier, queued)
            if idempotency_key is not None:
                self.keys[idempotency_key] = digest, identifier
        return dict(job_id=identifier, status="queued")

    def _run(self, identifier, queued):
        begin = time.monotonic()
        with self.lock:
            self.worker_started[identifier] = begin
        initialization = 0.
        try:
            if not self.runtime.value:
                started = time.monotonic()
                devices = (C.c_uint*len(self.devices))(*self.devices)
                result = self.lib.odz_runtime_create_devices(devices, len(devices), self.cache.encode(), C.byref(self.runtime))
                initialization = time.monotonic()-started
                if result:
                    message = self.lib.odz_runtime_error(self.runtime) or b"CUDA runtime initialization failed"
                    self.lib.odz_runtime_destroy(self.runtime)
                    self.runtime = C.c_void_p()
                    raise RuntimeError(message.decode(errors="replace"))
            self.lib.odz_job_run(self.runtime, self.handles[identifier])
        except Exception as exc:
            with self.lock:
                self.failures[identifier] = str(exc)
        try:
            result = self.results(identifier)
        except Exception as exc:
            with self.lock:
                self.failures[identifier] = str(exc)
            result = dict(job_id=identifier, status="failed", error=str(exc))
        with self.lock:
            self.transport[identifier] = dict(queue_seconds=begin-queued,
                runtime_initialization_seconds=initialization, worker_seconds=time.monotonic()-begin)
            result["transport_timing"] = self.transport[identifier]
            self.completed[identifier] = time.monotonic()
        return result

    def _report(self, identifier, *, summary=False):
        with self.lock:
            handle = self.handles[identifier]
            function = self.lib.odz_job_status if summary else self.lib.odz_job_report
            required = C.c_size_t()
            if function(handle, None, 0, C.byref(required)):
                raise RuntimeError("Could not measure native report")
            while True:
                output = C.create_string_buffer(required.value)
                code = function(handle, output, len(output), C.byref(required))
                if code == 1:
                    continue
                if code:
                    raise RuntimeError("Could not serialize native report")
                result = json.loads(output.value)
                break
            if identifier in self.failures:
                result.update(status="failed", error=self.failures[identifier])
            if summary and identifier in self.worker_started:
                result["timing"]["worker_seconds"] = self.transport.get(identifier,{}).get(
                    "worker_seconds",time.monotonic()-self.worker_started[identifier])
                if result["status"]=="queued":result["status"]="initializing"
            if identifier in self.transport:
                result["transport_timing"] = dict(self.transport[identifier])
        result["job_id"] = identifier
        return result

    def status(self, job_id):
        return self._report(job_id, summary=True)

    def results(self, job_id, format="compact"):
        if format not in ("compact", "full"):
            raise ValueError("format must be compact or full")
        result = self._report(job_id)
        from .reference import expression
        for candidate in result["candidates"].values():
            candidate["equations"] = dict(zip(result["states"],
                (expression(p, result["states"]) for p in candidate["resolved_programs"])))
        # Both formats are normalized; no repeated candidate bodies per leaderboard.
        return result

    def cancel(self, job_id):
        with self.lock:
            self.lib.odz_job_cancel(self.handles[job_id])
        return dict(job_id=job_id, cancellation_requested=True)

    def replay(self, job_id, candidate_id=None, *, address=None, cpu=False):
        if (candidate_id is None) == (address is None):
            raise ValueError("Provide exactly one of candidate_id or address")
        report = self.results(job_id)
        if address is not None:
            candidate_id = ":".join(str(address[x]) for x in
                                    ("family_index", "ast_index", "bank_index", "permutation"))
        candidate = deepcopy(report["candidates"][candidate_id])
        result = dict(job_id=job_id, candidate_id=candidate_id, candidate=candidate,
                      sampling_profile=report["sampling_profile"], rng_profile=report["rng_profile"])
        if cpu:
            from .problem import prepare_problem, cpu_score
            problem = prepare_problem(json.loads(self.requests[job_id])["problem"])
            result["cpu_reference"] = cpu_score(problem, candidate["resolved_programs"],
                steps=report["integration"]["steps_per_observation"], breakdown=True)
        return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stopping.set()
            for handle in self.handles.values():
                self.lib.odz_job_cancel(handle)
        # Destroy on the same worker after all submitted jobs have stopped.
        def destroy():
            self.lib.odz_runtime_destroy(self.runtime)
            with self.lock:
                for handle in self.handles.values():
                    self.lib.odz_job_destroy(handle)
                self.handles.clear()
                for table in (self.jobs,self.requests,self.failures,self.keys,self.problems,
                              self.transport,self.completed,self.problem_times,self.problem_sizes,self.worker_started):
                    table.clear()
        self.executor.submit(destroy).result()
        self.executor.shutdown()
        self.reaper.join()
