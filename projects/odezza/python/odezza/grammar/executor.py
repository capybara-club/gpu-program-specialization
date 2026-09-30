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
"""Persistent native execution, bounded numeric chunks, and separate native LM."""
from __future__ import annotations
from collections import OrderedDict
import ctypes as C
import hashlib
import math
import os
from pathlib import Path
import struct
import time

from odegrammar.compiler import content_id, compile_lm_request
from odezza.native import AstProgram, LmFit, LmPipeline, LmShape
from . import VERSION
from .cuda import CUDA, D, U, nvrtc_version
from .lowering import CompatibilityError, Layout, f32, identities, lower
from .native_scoring import Launch, Pipeline, Reducer, Template, library
from .prelude import Prelude, PROFILE


def floats(values):
    return struct.pack("<"+"f"*len(values), *values)


class Executor:
    def __init__(self, library_path, device=0):
        self.library_path = str(Path(library_path).resolve(strict=True))
        self.library_sha256 = hashlib.sha256(Path(self.library_path).read_bytes()).hexdigest()
        if os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER") != "EAGER":
            raise CompatibilityError("Set CUDA_MODULE_LOADING=EAGER before starting the grammar worker")
        self.gpu = CUDA(device)
        self.lib = library(self.library_path)
        self.pipelines, self.lm_pipelines = OrderedDict(), OrderedDict()
        self.templates = OrderedDict()
        self.prelude, self.reducer = None, None
        self.closed = False

    @property
    def profile(self):
        if hasattr(self, "_profile"):
            return self._profile
        def source_hash(path):
            digest = hashlib.sha256()
            for source in sorted(path.glob("*.py")):
                digest.update(source.name.encode()); digest.update(source.read_bytes())
            return digest.hexdigest()
        import odegrammar
        self._profile = dict(adapter=VERSION, native_library_sha256=self.library_sha256,
                    adapter_source_sha256=source_hash(Path(__file__).parent),
                    compiler_source_sha256=source_hash(Path(odegrammar.__file__).parent),
                    cuda_driver_version=self.gpu.driver_version, nvrtc_version=nvrtc_version(),
                    sm=self.gpu.sm, rng=PROFILE, precision="float32", rhs_abi="odezza-postorder-f32-v2")
        return self._profile

    def initialize(self, max_bank_bytes):
        if self.prelude is None: self.prelude = Prelude(self.gpu, max_bank_bytes)
        self.prelude.max_bank_bytes = max_bank_bytes
        if self.reducer is None: self.reducer = Reducer(self.lib, self.gpu)

    def data(self, problem):
        reference = [row[s] if row[s] is not None else math.nan for s in range(len(problem["states"])) for row in problem["rows"]]
        arrays = [struct.pack("<"+"I"*len(problem["offsets"]), *problem["offsets"]), floats(problem["times"]), floats(reference)]
        result = []
        try:
            for data in arrays: result.append(self.gpu.buffer(len(data), data))
            self.gpu.fence()
            return result
        except BaseException:
            for b in reversed(result): b.close()
            raise

    def pipeline(self, nstates, nslots, options):
        key = (nstates, nslots, options["module_systems"], options["patch_capacity"])
        start = time.monotonic(); hit = key in self.pipelines
        if not hit:
            capacity = min(max(8, (nslots+7)//8*8), 253-2*nstates)
            template_key = (nstates, capacity, options["module_systems"], options["patch_capacity"])
            if template_key not in self.templates:
                self.templates[template_key] = Template(self.lib, self.gpu.sm, *template_key)
            self.templates.move_to_end(template_key)
            self.pipelines[key] = Pipeline(self.lib, self.gpu, *key, template=self.templates[template_key])
            while len(self.templates) > 8: self.templates.popitem(last=False)[1].close()
        self.pipelines.move_to_end(key)
        while len(self.pipelines) > 4: self.pipelines.popitem(last=False)[1].close()
        return self.pipelines[key], time.monotonic()-start, hit

    def score_batch(self, problem, integration, batch, options, keep_k, on_chunk, stop):
        """Each item contains skeleton, variant, layout and native programs."""
        start_time = time.monotonic()
        self.initialize(options["max_bank_bytes"])
        nslots = len(batch[0]["layout"].slots)
        layout = batch[0]["layout"]
        permutations, numeric = layout.permutation_count, layout.numeric_count
        nsystems = len(batch)
        if any((len(b["layout"].slots), b["layout"].permutation_count, b["layout"].numeric_count) != (nslots, permutations, numeric) for b in batch):
            raise ValueError("Incompatible batch layout")
        # Reserve room for raw scores, coefficient banks, and reduction tiles/winners.
        per_bank_bytes = nsystems*(4*permutations+4*nslots+64)
        overhead = nsystems*permutations*(16*(16+4*nslots)+24)
        if overhead >= options["max_device_bytes"]:
            raise CompatibilityError("Toggle product exceeds the configured device reduction budget")
        chunk = min(numeric, 2**32-1, max(1, options["max_chunk_configurations"]//(nsystems*permutations)),
                    (options["max_device_bytes"]-overhead)//max(1, per_bank_bytes))
        if chunk < 1:
            raise CompatibilityError("One bank exceeds the configured device memory budget")
        while True:
            reduction = self.reducer.requirements(Launch(bank_count=chunk, toggle_bits=layout.toggle_bits), nsystems)
            scratch_bytes = (nsystems*chunk*(nslots+permutations)*4 + reduction.workspace + reduction.winners +
                             reduction.counts + reduction.groups*self.reducer.k*nslots*4)
            if scratch_bytes <= options["max_device_bytes"]: break
            if chunk == 1:
                raise CompatibilityError("Exact native reduction workspace exceeds max_device_bytes")
            chunk = max(1, chunk//2)
        handle, prepare_seconds, cache_hit = self.pipeline(len(problem["states"]), nslots, options)
        resources = self.data(problem)
        descriptors = None
        try:
            rng_before = self.prelude.generation_seconds
            descriptors, owned = self.prelude.descriptors([b["variant"] for b in batch], [b["layout"] for b in batch])
            resources.extend(owned)
            coefficients = self.gpu.buffer(nsystems*chunk*nslots*4); resources.append(coefficients)
            scores = self.gpu.buffer(nsystems*chunk*permutations*4); resources.append(scores)
            self.gpu.fence()
            execution = content_id(dict(problem=problem["id"], integration=integration, profile=self.profile))
            for base in range(0, numeric, chunk):
                if stop(): break
                banks = min(chunk, numeric-base)
                prelude_seconds = self.prelude.run(descriptors, coefficients, nslots, nsystems, base, banks)
                launch = Launch(coefficients.ptr.value, banks, layout.toggle_bits,
                                resources[0].ptr.value, resources[1].ptr.value, resources[2].ptr.value,
                                len(problem["offsets"])-1, len(problem["times"]), integration["steps_per_observation"],
                                scores.ptr.value, 0, 0, 0, 0, 1, self.gpu.ready.value)
                self.gpu.record(self.gpu.ready)
                native = handle.run([b["programs"] for b in batch], launch)
                retained = [dict() for _ in range(nsystems*permutations)]
                valid = invalid = 0
                first = True; reduction_seconds = transfer_seconds = 0.; rounds = 0
                retention_complete = True
                while True:
                    if not first and stop():
                        retention_complete = False
                        break
                    before = time.monotonic()
                    winners, counts, values, seconds, outputs = self.reducer.run(launch, nsystems, nslots)
                    reduction_seconds += seconds
                    transfer_seconds += time.monotonic()-before-seconds
                    rounds += 1
                    try:
                        if first:
                            valid, invalid = sum(c.valid for c in counts), sum(c.invalid for c in counts)
                        for group, count in enumerate(counts):
                            item = batch[group//permutations]
                            for rank in range(self.reducer.k):
                                if len(retained[group]) >= keep_k: break
                                i = group*self.reducer.k+rank
                                winner = winners[i]
                                if winner.index == 2**64-1: continue
                                local = winner.index % (banks*permutations)
                                bank, permutation = divmod(local, permutations)
                                row = values[i*nslots:(i+1)*nslots]
                                numeric_id, structure_id, resolved = identities(item["skeleton"], item["variant"], problem["states"], item["layout"], row, permutation)
                                if numeric_id in retained[group] or len(retained[group]) >= keep_k: continue
                                index = item["layout"].grammar_index(base+bank, permutation)
                                candidate = dict(variant_id=item["variant"]["id"], skeleton_id=item["skeleton"]["id"],
                                    configuration_index=index, bank_index=base+bank, permutation=permutation,
                                    mse=winner.mse, values=row, slots=item["layout"].slots, numeric_id=numeric_id,
                                    structure_id=structure_id, resolved_programs=resolved, execution_id=execution,
                                    integration=integration, profile=self.profile, problem_id=problem["id"])
                                candidate["id"] = content_id(candidate)
                                retained[group][numeric_id] = candidate
                        needed = any(len(r) < keep_k and c.valid > self.reducer.k for r, c in zip(retained, counts))
                        if not needed: break
                        self.gpu.launch(self.prelude.consume, len(winners), [scores.ptr, outputs[1].ptr, D(len(winners))])
                        self.gpu.fence()
                        first = False
                    finally:
                        for output in reversed(outputs): output.close()
                profile = dict(native=native, template_prepare_seconds=prepare_seconds,
                    template_cache_hit=cache_hit, prelude_kernel_seconds=prelude_seconds,
                    rng_kernel_seconds=self.prelude.generation_seconds-rng_before,
                    reducer_kernel_seconds=reduction_seconds, reduction_transfer_host_seconds=transfer_seconds,
                    reducer_passes=rounds, bank_start=base, bank_count=banks,
                    retention_complete=retention_complete,
                    scratch_device_bytes=scratch_bytes,
                    valid=valid, invalid=invalid, configurations=nsystems*banks*permutations)
                on_chunk([candidate for group in retained for candidate in group.values()], profile)
                prepare_seconds = 0.; rng_before = self.prelude.generation_seconds
        finally:
            for resource in reversed(resources): resource.close()
            self.prelude.trim()
        return time.monotonic()-start_time

    def fit(self, problem, skeleton, variant, candidate, request, *, lanes=1):
        spec = compile_lm_request(request)
        if spec["settings"]["max_iterations"] > 2**32-1 or any(f32(spec["settings"][k]) <= 0 for k in ("damping", "tolerance")):
            raise CompatibilityError("LM controls exceed the native uint32/positive-FP32 contract")
        if spec["states"] != problem["states"]:
            raise CompatibilityError("LM state order differs from scored problem")
        if candidate["id"] not in spec["candidate_ids"]:
            raise ValueError("Candidate not requested")
        if "integration" in spec:
            from .problem import resolve_integration
            if resolve_integration(spec["integration"]["settings"], problem) != candidate["integration"]:
                raise CompatibilityError("LM must preserve the scored integration settings")
        layout = Layout.from_pools(variant["pools"])
        active = set(skeleton["active_slots"]["param"])
        if not set(spec["parameters"]) <= active:
            raise CompatibilityError("LM requested inactive or non-parameter slots")
        fixed = dict(zip(layout.slots, candidate["values"]))
        fitted = {("param", name): i for i, name in enumerate(spec["parameters"])}
        programs = [lower(skeleton["rhs"][s], variant["pools"], layout, fixed=fixed,
                          fitted=fitted, permutation=candidate["permutation"]) for s in problem["states"]]
        p, n = len(fitted), len(programs)
        key = n, p, lanes
        started = time.monotonic()
        if key not in self.lm_pipelines:
            self.lm_pipelines[key] = LmPipeline(self.library_path, self.gpu.sm, LmShape(n, p, 384, lanes),
                                               fallback_lanes=tuple(w for w in (2, 4, 8) if w > lanes))
        handle = self.lm_pipelines[key]
        self.lm_pipelines.move_to_end(key)
        while len(self.lm_pipelines) > 4: self.lm_pipelines.popitem(last=False)[1].close()
        resources = self.data(problem)
        try:
            arrays = []
            def buffer(raw):
                b = self.gpu.buffer(len(raw), raw); resources.append(b); return b.ptr.value
            reference = [row[s] if row[s] is not None else 0. for s in range(n) for row in problem["rows"]]
            weights = [float(row[s] is not None) for s in range(n) for row in problem["rows"]]
            fit = LmFit(); fit.offsets_device, fit.times_device = resources[0].ptr.value, resources[1].ptr.value
            fit.reference_device, fit.weights_device = buffer(floats(reference)), buffer(floats(weights))
            for code in programs: arrays.append(C.create_string_buffer(code))
            asts = (AstProgram*n)(*[AstProgram(C.addressof(b), len(code)) for b, code in zip(arrays, programs)])
            fit.rhs = asts
            fit.starts_device = buffer(floats([fixed[pair] for pair in fitted])); fit.start_count = 1
            fit.lower_device = buffer(floats([-3e38]*p)); fit.upper_device = buffer(floats([3e38]*p))
            fit.trajectory_count, fit.point_count = len(problem["offsets"])-1, len(problem["times"])
            fit.steps_per_interval = candidate["integration"]["steps_per_observation"]
            fit.max_iterations = spec["settings"]["max_iterations"]; fit.max_damping_attempts = 8
            fit.initial_damping = spec["settings"]["damping"]; fit.max_step = 3e38
            fit.target_mse = spec["settings"]["tolerance"]
            outputs = []
            for field, size in (("parameters_device", p*4), ("initial_mse_device", 4), ("mse_device", 4),
                                *[(name, 4) for name in ("iterations_device", "accepted_device", "factorizations_device", "evaluations_device", "invalid_device")]):
                b = self.gpu.buffer(size); resources.append(b); outputs.append(b); setattr(fit, field, b.ptr.value)
            self.gpu.record(self.gpu.ready); fit.input_ready_event = self.gpu.ready.value
            try: native = handle.run([fit])
            except BaseException:
                if handle.fatal: self.gpu.fatal = True
                raise
            parameters = list(struct.unpack("<"+"f"*p, outputs[0].read()))
            initial, final = [struct.unpack("<f", b.read())[0] for b in outputs[1:3]]
            counts = [struct.unpack("<I", b.read())[0] for b in outputs[3:]]
            fixed.update(dict(zip(fitted, parameters)))
            result = dict(candidate)
            for display_field in ("equations", "named_values", "cpu_reference"):
                result.pop(display_field, None)
            result["parent_candidate_id"] = candidate["id"]
            result["values"] = [fixed[pair] for pair in layout.slots]
            result["numeric_id"], result["structure_id"], result["resolved_programs"] = identities(skeleton, variant, problem["states"], layout, result["values"], candidate["permutation"])
            result.update(mse=final if math.isfinite(final) and final < 3e38 else None,
                          initial_mse=initial if math.isfinite(initial) and initial < 3e38 else None,
                          fitted_parameters=dict(zip(spec["parameters"], parameters)),
                          lm=dict(request=spec, native=native, iterations=counts[0], accepted_steps=counts[1],
                                  factorizations=counts[2], evaluations=counts[3], invalid=counts[4],
                                  tolerance_semantics="target_observed_mse", seconds=time.monotonic()-started))
            result["fit_status"] = ("invalid" if result["mse"] is None else "target_mse" if final <= fit.target_mse
                                    else "iteration_limit" if counts[0] >= fit.max_iterations else "stopped")
            result.pop("id", None); result["id"] = content_id(result)
            return result
        finally:
            for resource in reversed(resources): resource.close()

    def close(self):
        if self.closed: return
        for p in self.lm_pipelines.values(): p.close()
        for p in self.pipelines.values(): p.close()
        for p in self.templates.values(): p.close()
        if self.reducer: self.reducer.close()
        if self.prelude: self.prelude.close()
        self.gpu.close(); self.closed = True
