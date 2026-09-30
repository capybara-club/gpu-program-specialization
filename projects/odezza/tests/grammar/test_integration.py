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
import copy
import ctypes as C
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from odegrammar.compiler import Compiler, evaluate_postorder
from odegrammar.pools import configuration_indices, evaluate_prelude
from odezza.grammar.lowering import CompatibilityError, Layout, lower_system, identities
from odezza.grammar.reference import evaluate, expression
from odezza.grammar.problem import prepare_problem, resolve_integration
from odezza.grammar.prelude import Slot, bank_address
from odezza.grammar.mcp import Transport
from odezza.grammar.service import Service
from odezza.grammar.native_scoring import Pipeline
from odezza.grammar.retention import Ranking, ranked


ROOT = Path(__file__).resolve().parents[2]


def problem():
    times = [0, .03, .1, .23]
    return dict(states=["x0", "x1"], trajectories=[dict(times=times, initial=[1, 2],
                values=[[math.exp(-t), None] for t in times])])


def grammar():
    return dict(version=1, states=["x0", "x1"], integration=dict(method="rk4", dt=.01),
                rhs=dict(x0="-const.k*leaf.s", x1="0"),
                constants=dict(k=dict(values=[.5, 1., 2.])),
                leaves=dict(s=dict(states=["x0", "x1"], arity=2)),
                retain={"global": {"k": 4, "unit": "numeric_candidate"}, "per_family": 3})


class IntegrationTests(unittest.TestCase):
    def test_native_lowering_matches_compiler_all_product_indices(self):
        request = json.loads((ROOT/"examples/grammar/constant_rng_product.json").read_text())
        request["leaves"] = {"a": {"states": request["states"], "arity": 2}, "b": {"states": request["states"], "arity": 2}}
        request["rhs"]["x0"] = "leaf.a-2*leaf.b+pow(x0,3)+pow(x1,-2)"
        records = list(Compiler(request).records())
        skeleton = next(r for r in records if r["type"] == "skeleton")
        variant = next(r for r in records if r["type"] == "variant")
        layout, programs = lower_system(skeleton, variant, request["states"])
        for index in range(variant["pools"]["configuration_count"]):
            prepared = evaluate_prelude(variant["pools"], configuration_indices(variant["pools"], index))
            values = [prepared[namespace][name] for namespace, name in layout.slots]
            native = layout.native_index(index)
            bank, permutation = divmod(native, layout.permutation_count)
            self.assertEqual(layout.grammar_index(bank, permutation), index)
            for state, program in zip(request["states"], programs):
                expected = evaluate_postorder(skeleton["rhs"][state], [1.2, .7], constants=prepared["const"],
                    random_values=prepared["rng"], parameters=prepared["param"],
                    toggles=variant["pools"]["toggles"], choices={k: (permutation >> b[0]) & ((1 << len(b))-1) for k, b in layout.bits.items()})
                self.assertAlmostEqual(evaluate(program, [1.2, .7], values, permutation), expected, places=6)

    def test_readable_equation_preserves_operator_order(self):
        request = grammar(); request["rhs"]["x0"] = "x0-(x1+x0)/(1+x1)"
        rows = list(Compiler(request).records())
        s = next(x for x in rows if x["type"] == "skeleton"); v = next(x for x in rows if x["type"] == "variant")
        _, programs = lower_system(s, v, request["states"])
        self.assertEqual(expression(programs[0], request["states"]), "x0 - (x1 + x0) / (1 + x1)")

    def test_mask_initials_and_dt(self):
        p = prepare_problem(problem())
        self.assertEqual(p["observed_scalars"], 3)
        self.assertEqual(p["rows"][0], [1., 2.])
        self.assertTrue(all(row[1] is None for row in p["rows"][1:]))
        settings = resolve_integration({"dt": .01}, p)
        self.assertGreaterEqual(settings["steps_per_observation"], 13)
        with self.assertRaises(CompatibilityError): resolve_integration({"dt": .01, "max_steps": 1}, p)
        bad = problem(); bad["trajectories"][0]["initial"] = [1]
        with self.assertRaises(ValueError): prepare_problem(bad)
        bad = problem(); bad["trajectories"][0]["values"][1][0] = math.inf
        with self.assertRaises(ValueError): prepare_problem(bad)

    def test_production_bank_identity_excludes_length(self):
        r = json.loads((ROOT/"examples/grammar/constant_rng_product.json").read_text())
        records = list(Compiler(r).records())
        banks = next(v for v in records if v["type"] == "variant")["pools"]["rng_bank_requests"]
        for bank in banks:
            self.assertEqual(bank_address(bank), bank_address(dict(bank, count=1000)))
        self.assertEqual(C.sizeof(Slot), 48)

    def test_preflight_rejects_time_before_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Service(directory)
            try:
                p = service.prepare(problem())
                request = grammar(); request["rhs"]["x0"] = "t"
                job = service.submit(p["problem_id"], request, plan_only=True)
                report = service.jobs[job["job_id"]].result()
                self.assertEqual(report["status"], "failed")
                self.assertIn("TIME", report["error"])
                self.assertEqual(report["counts"]["completed_configurations"], 0)
                self.assertIsNone(service.executor)
            finally: service.close()

    def test_plan_identity_partial_and_mcp(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Service(directory)
            try:
                p = service.prepare(problem())
                first = service.submit(p["problem_id"], grammar(), idempotency_key="same", plan_only=True)
                second = service.submit(p["problem_id"], grammar(), idempotency_key="same", plan_only=True)
                self.assertEqual(first["job_id"], second["job_id"])
                report = service.jobs[first["job_id"]].result()
                self.assertEqual(report["generation"]["configuration_visits"], 6)
                self.assertFalse(report["gpu_executed"])
                transport = Transport(service)
                self.assertIn("error", transport.dispatch(dict(jsonrpc="2.0", id=1, method="tools/list")))
                init = transport.dispatch(dict(jsonrpc="2.0", id=2, method="initialize", params={"protocolVersion": "2025-06-18"}))
                self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
                transport.dispatch(dict(jsonrpc="2.0", method="notifications/initialized"))
                response = transport.dispatch(dict(jsonrpc="2.0", id=3, method="tools/call", params={"name": "odezza_status", "arguments": {"job_id": first["job_id"]}}))
                self.assertEqual(response["result"]["structuredContent"]["status"], "planned")
            finally: service.close()

    def test_fp32_rng_bounds_and_execution_budget_rejected_before_gpu(self):
        request = grammar(); request.pop("constants"); request.pop("leaves")
        request["rhs"]["x0"] = "rng.a*x0"
        request["rng_banks"] = {"u": {"base": "uniform01", "count": 4}}
        request["rng"] = {"a": {"bank": "u", "transform": {"kind": "uniform", "low": 1., "high": 1.+1e-10}}}
        records = list(Compiler(request).records())
        skeleton = next(r for r in records if r["type"] == "skeleton")
        variant = next(r for r in records if r["type"] == "variant")
        with self.assertRaises(CompatibilityError): lower_system(skeleton, variant, request["states"])
        with tempfile.TemporaryDirectory() as directory:
            service = Service(directory)
            try:
                with self.assertRaises(RuntimeError): Service(directory)
                prepared = service.prepare(problem())
                job = service.submit(prepared["problem_id"], grammar(), {"max_chunk_configurations": 1}, plan_only=True)
                result = service.jobs[job["job_id"]].result()
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["counts"]["completed_configurations"], 0)
                self.assertIn("toggle product", result["error"])
            finally: service.close()

    def test_structure_identity_preserves_sharing_and_ignores_slot_spelling(self):
        def identity(name):
            r = grammar(); r.pop("leaves"); r["constants"] = {name: {"value": 1.}}
            r["rhs"]["x0"] = f"const.{name}*x0+const.{name}*x1"
            rows = list(Compiler(r).records())
            s = next(x for x in rows if x["type"] == "skeleton"); v = next(x for x in rows if x["type"] == "variant")
            layout, _ = lower_system(s, v, r["states"])
            return identities(s, v, r["states"], layout, [1.], 0)
        self.assertEqual(identity("a")[:2], identity("z")[:2])

    def test_native_scoring_releases_handle_after_workspace_failure(self):
        destroyed = []
        def create(info, out): out._obj.value = 123; return 0
        lib = SimpleNamespace(odezza_scoring_pipeline_create=create,
                              odezza_scoring_pipeline_workspace_requirements=lambda *args: 1,
                              odezza_scoring_pipeline_write_error=lambda *args: 0,
                              odezza_scoring_pipeline_destroy=lambda handle: destroyed.append(handle.value) or 0)
        with self.assertRaises(RuntimeError): Pipeline(lib, SimpleNamespace(sm=120, fatal=False), 2, 1)
        self.assertEqual(destroyed, [123])

    def test_streaming_retention_matches_full_sort_with_late_better_duplicates(self):
        rows = [dict(id=str(i), mse=float((i*37)%31), variant_id=str(i%13), configuration_index=i,
                     numeric_id=str(i%17), structure_id=str(i%7)) for i in range(200)]
        for unit in ("numeric_candidate", "resolved_structure", "variant"):
            for k in (0, 1, 5, 20):
                policy = dict(unit=unit, k=k); ranking = Ranking(policy)
                for row in reversed(rows): ranking.offer(row)
                self.assertEqual(ranking.results(), ranked(rows, policy))


if __name__ == "__main__": unittest.main()
