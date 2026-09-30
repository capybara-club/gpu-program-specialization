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
"""Focused native execution acceptance. No CPU scoring fallback is permitted."""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import struct
import tempfile
from unittest.mock import patch

from odegrammar.compiler import Compiler
from odegrammar.pools import evaluate_prelude, configuration_indices
from odezza.grammar.lowering import Layout, lower_system
from odezza.grammar.service import execution_options
from odezza.grammar.problem import resolve_integration
from odezza.grammar.service import Service, atomic
from odezza.grammar.prelude import bank_address
from odezza.grammar.problem import cpu_score
from test_integration import problem, grammar, ROOT


def main():
    p = argparse.ArgumentParser(); p.add_argument("--library", required=True); p.add_argument("--root", required=True)
    args = p.parse_args(); service = Service(args.root, library=args.library)
    results = []
    try:
        prepared = service.prepare(problem())
        def run(name, request, execution=None):
            job = service.submit(prepared["problem_id"], request, execution)
            report = service.jobs[job["job_id"]].result()
            print(name, json.dumps({k: report.get(k) for k in ("status", "error", "counts", "timing")}), flush=True)
            assert report["status"] == "complete", report
            expected = list(Compiler(request).records())[-1]["configuration_visits"]
            assert report["counts"]["completed_configurations"] == expected, report
            for candidate in report["winners"].get("global", []):
                replay = service.replay(job["job_id"], candidate["id"], cpu=True)
                cpu = replay["cpu_reference"]["mse"]
                assert cpu is not None and abs(cpu-candidate["mse"]) < 3e-5*max(1., cpu), (cpu, candidate)
            compact = service.results(job["job_id"], format="compact")
            assert compact["leaderboards"].get("global", []) == [r["id"] for r in report["winners"].get("global", [])]
            for candidate in compact["candidates"].values():
                direct = service.replay(job["job_id"], candidate["id"])
                indexed = service.replay(job["job_id"], address=candidate["origin"])
                assert (direct["mse"], direct["values"], direct["resolved_programs"]) == (indexed["mse"], indexed["values"], indexed["resolved_programs"])
            results.append(dict(name=name, job_id=job["job_id"], **report))
            return job, report
        request = grammar()
        job, report = run("masked_irregular_toggle", request)
        best = report["winners"]["global"][0]
        assert best["mse"] < 1e-10, best
        assert best["values"] == [1.] and best["permutation"] == 0, best
        _, chunked = run("chunk_invariance", request, {"max_chunk_configurations": 2})
        assert [(r["numeric_id"], r["mse"]) for r in report["winners"]["global"]] == [(r["numeric_id"], r["mse"]) for r in chunked["winners"]["global"]]
        product = json.loads((ROOT/"examples/grammar/constant_rng_product.json").read_text())
        job, product_report = run("mixed_rng_constant_product", product)
        inline = service.submit(problem=problem(), grammar=product)
        inline_report = service.jobs[inline["job_id"]].result()
        assert inline_report["status"] == "complete", inline_report
        assert inline_report["winners"] == product_report["winners"]
        results.append(dict(name="inline_problem_matches_prepared_rng_job", job_id=inline["job_id"], **inline_report))
        _, product_chunked = run("rng_chunk_invariance", product, {"max_chunk_configurations": 17})
        assert [(r["numeric_id"], r["mse"]) for r in product_report["winners"]["global"]] == [(r["numeric_id"], r["mse"]) for r in product_chunked["winners"]["global"]]
        def check_transforms():
            request = json.loads(json.dumps(product))
            request["rng"].update(identity={"bank": "u", "axis": "trial", "transform": {"kind": "identity"}},
                                  log={"bank": "u", "axis": "trial", "transform": {"kind": "log_uniform", "low": .01, "high": "const.scale"}})
            request["rhs"]["x1"] += "+rng.identity+rng.log"
            variant = next(r for r in Compiler(request).records() if r["type"] == "variant")
            layout = Layout.from_pools(variant["pools"]); executor = service.gpu_executor()
            descriptors, resources = executor.prelude.descriptors([variant], [layout])
            output = executor.gpu.buffer(layout.numeric_count*len(layout.slots)*4); resources.append(output)
            try:
                executor.prelude.run(descriptors, output, len(layout.slots), 1, 0, layout.numeric_count)
                actual = struct.unpack("<"+"f"*(layout.numeric_count*len(layout.slots)), output.read())
                raw = {bank["key"]: struct.unpack("<"+"f"*bank["count"], executor.prelude.bank(bank).read(bank["count"]*4))
                       for bank in variant["pools"]["rng_bank_requests"]}
                with patch("odegrammar.pools.bank_value", side_effect=lambda bank, index: raw[bank["key"]][index]):
                    for index in range(layout.numeric_count):
                        expected = evaluate_prelude(variant["pools"], configuration_indices(variant["pools"], index))
                        for slot, pair in enumerate(layout.slots):
                            value = expected[pair[0]][pair[1]]
                            observed = actual[index*len(layout.slots)+slot]
                            assert abs(value-observed) < 2e-6*max(1., abs(value)), (index, pair, value, observed)
                return layout.numeric_count
            finally:
                for resource in reversed(resources): resource.close()
        assert service.worker.submit(check_transforms).result() == 384
        # Exact uniform conformance with the already-tested native Philox CPU primitive.
        def check_uniform():
            executor = service.gpu_executor()
            bank = next(r for r in list(Compiler(product).records()) if r["type"] == "variant")["pools"]["rng_bank_requests"]
            lib = executor.lib
            lib.odezza_rng_uniform.argtypes = [C.c_uint64]*3+[C.POINTER(C.c_float)]
            lib.odezza_rng_uniform.restype = C.c_int
            checked = 0
            for request in bank:
                if request["base"] != "uniform01": continue
                buffer = executor.prelude.bank(request)
                values = struct.unpack("<"+"f"*request["count"], buffer.read(request["count"]*4))
                seed, stream = bank_address(request)
                for i, value in enumerate(values):
                    expected = C.c_float(); assert lib.odezza_rng_uniform(seed, stream, i, C.byref(expected)) == 0
                    assert value == expected.value, (i, value, expected.value)
                    checked += 1
            return checked
        assert service.worker.submit(check_uniform).result() == 20
        request = grammar(); request.pop("leaves"); request["rhs"]["x0"] = "-const.k*x0"
        request["constants"]["k"]["values"] = [float(i)/20 for i in range(1, 42)]
        request["families"] = [{"id": "first", "tags": ["early"]}, {"id": "second", "tags": ["late"]}]
        request["retain"] = {"global": 20, "per_family": 20, "by_tag": {"late": 20}}
        _, report = run("late_provenance_top20", request)
        assert len(report["winners"]["global"]) == 20
        assert report["winners"]["families"]["first"] == report["winners"]["families"]["second"]
        assert len(report["winners"]["tags"]["late"]) == 20
        request["constants"]["k"]["values"] = [1.]*35+[2.]
        request["retain"] = {"global": 2, "per_family": 2}
        _, duplicates = run("duplicate_rows_do_not_fill_topk", request)
        assert len(duplicates["winners"]["global"]) == 2
        assert {r["values"][0] for r in duplicates["winners"]["global"]} == {1., 2.}
        def cancel_during_reduction():
            records = list(Compiler(request).records())
            s = next(r for r in records if r["type"] == "skeleton")
            v = next(r for r in records if r["type"] == "variant")
            layout, programs = lower_system(s, v, request["states"])
            prepared_problem = service.problem(prepared["problem_id"])
            calls, profiles = [], []
            def stop():
                calls.append(1)
                return len(calls) > 1
            service.gpu_executor().score_batch(prepared_problem, resolve_integration(request["integration"], prepared_problem),
                [dict(skeleton=s, variant=v, layout=layout, programs=programs)], execution_options(), 2,
                lambda rows, profile: profiles.append(profile), stop)
            assert profiles[0]["configurations"] == 36 and not profiles[0]["retention_complete"], profiles
        service.worker.submit(cancel_during_reduction).result()
        request = grammar(); request.pop("constants"); request.pop("leaves")
        request["parameters"] = {"k": {"initial": .5}}
        request["rhs"]["x0"] = "-theta.k*x0"
        job, report = run("lm_screen", request)
        identifier = report["winners"]["global"][0]["id"]
        fitted = service.fit(job["job_id"], dict(version=1, kind="lm", states=request["states"], candidate_ids=[identifier],
                                               parameters=["k"], settings={"max_iterations": 30, "tolerance": 1e-10})).result()
        print("LM", json.dumps(fitted), flush=True)
        result = fitted["candidates"][0]
        assert abs(result["fitted_parameters"]["k"]-1) < 1e-3 and result["mse"] < 1e-9, result
        reference = cpu_score(service.problem(prepared["problem_id"]), result["resolved_programs"], steps=result["integration"]["steps_per_observation"])
        assert reference is not None and reference < 1e-9, reference
        replay_fit = service.replay(job["job_id"], result["id"], cpu=True)
        assert replay_fit["values"] == result["values"]
        results.append(dict(name="separate_native_lm", result=result, cpu_mse=reference))
        # Positive log correction must preserve every sampled/fixed scalar while fitting theta.
        request["rng_banks"] = {"u": {"base": "uniform01", "count": 8, "seed": 12}}
        request["rng"] = {"rate": {"bank": "u", "transform": {"kind": "uniform", "low": .2, "high": .8}}}
        request["constants"] = {"fixed": {"value": 1.}}
        request["parameters"] = {"log_k": {"initial": 0.}}
        request["rhs"]["x0"] = "-const.fixed*rng.rate*exp(theta.log_k)*x0"
        job, report = run("positive_log_lm_screen", request)
        parent = report["winners"]["global"][0]
        fitted = service.fit(job["job_id"], dict(version=1, kind="lm", states=request["states"], candidate_ids=[parent["id"]],
                            parameters=["log_k"], settings={"max_iterations": 40, "tolerance": 1e-10})).result()["candidates"][0]
        for slot, before, after in zip(parent["slots"], parent["values"], fitted["values"]):
            if slot[0] != "param": assert before == after, slot
        assert fitted["mse"] < 1e-9, fitted
        results.append(dict(name="native_lm_preserves_rng", result=fitted))
        invalid = grammar(); invalid["rhs"]["x0"] = "log(-x0)"; invalid.pop("constants"); invalid.pop("leaves")
        _, report = run("domain_failure_is_invalid", invalid)
        assert report["counts"]["valid"] == 0 and report["counts"]["invalid"] == 1
        assert report["winners"]["global"] == []
        states = [f"x{i}" for i in range(8)]
        times = [0., .03, .1, .2]
        rates = [.7+.1*i for i in range(8)]
        p8 = service.prepare(dict(states=states, trajectories=[dict(times=times, initial=[1.]*8,
                             values=[[math.exp(-t*k) for k in rates] for t in times])]))
        r8 = dict(version=1, states=states, integration={"method": "rk4", "dt": .01},
                  parameters={f"k{i}": {"initial": 1.} for i in range(8)},
                  rhs={s: f"-theta.k{i}*{s}" for i, s in enumerate(states)}, retain={"global": 1})
        job8 = service.submit(p8["problem_id"], r8)
        screen8 = service.jobs[job8["job_id"]].result()
        assert screen8["status"] == "complete", screen8
        fit8 = service.fit(job8["job_id"], dict(version=1, kind="lm", states=states,
            candidate_ids=[screen8["winners"]["global"][0]["id"]], parameters=[f"k{i}" for i in range(8)],
            settings={"max_iterations": 30, "tolerance": 1e-9})).result()["candidates"][0]
        assert fit8["mse"] < 1e-8, fit8
        assert max(abs(fit8["fitted_parameters"][f"k{i}"]-rates[i]) for i in range(8)) < .002, fit8
        results.append(dict(name="eight_state_eight_parameter_native_lm", result=fit8))
        atomic(Path(args.root)/"conformance.json", dict(status="passed", cases=results))
    finally: service.close()


if __name__ == "__main__": main()
