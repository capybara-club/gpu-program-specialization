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
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from odegrammar.compiler import Compiler, CompileError, SQLiteDedup, compile_lm_request, evaluate_postorder
from odegrammar.pools import configuration_indices, evaluate_prelude


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return {"version": 1, "states": ["x0", "x1"], "integration": {"method": "rk4", "dt": 0.01}, "rhs": {"x0": "x1", "x1": "-x0"}}


def collect(request, **kwargs):
    compiler = Compiler(request, **kwargs)
    records = list(compiler.records())
    return records, compiler.summary


def records_of(records, kind):
    return [r for r in records if r["type"] == kind]


class CompilerTests(unittest.TestCase):
    def test_postorder_evaluates_expected_rhs_and_stack_order(self):
        request = fixture()
        request["rhs"]["x1"] = "2*x0 - sin(x1)/(1+pow(x0,2)) + t"
        records, summary = collect(request)
        program = records_of(records, "skeleton")[0]["rhs"]["x1"]
        actual = evaluate_postorder(program, [0.5, 1.2], t=0.3)
        self.assertAlmostEqual(actual, 1.0 - math.sin(1.2)/1.25 + 0.3)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["unique_skeletons"], 1)

    def test_exhaustive_toggle_coverage_compiled_program(self):
        request = fixture()
        request["states"] = ["x0", "x1", "x2", "x3", "x4"]
        request["rhs"] = {s: "0" for s in request["states"]}
        request["rhs"]["x0"] = "leaf.a + 10*leaf.b"
        request["leaves"] = {
            "a": {"states": request["states"], "arity": 2, "coverage": "all"},
            "b": {"states": request["states"], "arity": 4, "coverage": "all"},
        }
        records, summary = collect(request)
        self.assertEqual(summary["unique_skeletons"], 1)
        self.assertEqual(summary["unique_variants"], 50)
        self.assertEqual(summary["configuration_visits"], 400)
        observed = set()
        for variant in records_of(records, "variant"):
            pool = variant["pools"]
            for index in range(pool["configuration_count"]):
                bindings = evaluate_prelude(pool, configuration_indices(pool, index))
                observed.add((bindings["leaf"]["a"], bindings["leaf"]["b"]))
        self.assertEqual(observed, set(itertools.product(range(5), repeat=2)))

    def test_tags_survive_dedup_and_subtree_spans(self):
        request = fixture()
        request["rules"] = {"R(z)": [{"expr": "sin(z)", "tags": ["mechanism_a"]}, {"expr": "sin(z)", "tags": ["mechanism_b"]}]}
        request["rhs"]["x1"] = "1+hole(R,x0)"
        records, summary = collect(request)
        self.assertEqual(summary["unique_skeletons"], 1)
        self.assertEqual(summary["unique_variants"], 1)
        self.assertEqual(summary["provenance_events"], 1)
        memberships = records_of(records, "variant") + records_of(records, "provenance")
        self.assertEqual(set(itertools.chain.from_iterable(m["tags"] for m in memberships)), {"mechanism_a", "mechanism_b"})
        for membership in memberships:
            self.assertEqual(membership["annotations"][0]["start"], 1)
            self.assertEqual(membership["annotations"][0]["end"], 3)

    def test_families_share_program_but_retain_different_numeric_domains(self):
        request = fixture()
        request["rhs"]["x1"] = "const.k*x0"
        request["families"] = [{"id": "small", "constants": {"k": {"values": [1, 2]}}}, {"id": "large", "constants": {"k": {"values": [3, 4]}}}]
        records, summary = collect(request)
        self.assertEqual(summary["unique_skeletons"], 1)
        self.assertEqual(summary["unique_variants"], 2)
        self.assertEqual(summary["configuration_visits"], 4)
        self.assertEqual({v["family_id"] for v in records_of(records, "variant")}, {"small", "large"})

    def test_families_identical_candidates_emit_membership(self):
        request = fixture()
        request["families"] = [{"id": "first"}, {"id": "second"}]
        records, summary = collect(request)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["unique_variants"], 1)
        self.assertEqual(records_of(records, "provenance")[0]["family_id"], "second")

    def test_local_rng_ranges_share_program_and_run_bank(self):
        request = fixture()
        request["rng_banks"] = {"u": {"base": "uniform01", "count": 4, "seed": 7, "scope": "run"}}
        request["rules"] = {"R(z)": [
            {"expr": "rng.a*z", "locals": {"rng": {"a": {"bank": "u", "transform": {"kind": "uniform", "low": low, "high": low+1}}}}}
            for low in (0, 10)
        ]}
        request["rhs"]["x1"] = "hole(R,x0)"
        records, summary = collect(request)
        self.assertEqual(summary["unique_skeletons"], 1)
        variants = records_of(records, "variant")
        self.assertEqual(len(variants), 2)
        bank_ids = [v["pools"]["rng_bank_requests"][0]["id"] for v in variants]
        self.assertEqual(bank_ids[0], bank_ids[1])
        values = [next(iter(evaluate_prelude(v["pools"])["rng"].values())) for v in variants]
        self.assertAlmostEqual(values[1] - values[0], 10)

    def test_local_constant_dependency_is_hygienic_and_cartesian(self):
        request = fixture()
        request["rng_banks"] = {"u": {"base": "uniform01", "count": 3, "seed": 2}}
        request["rules"] = {"R(z)": [{"expr": "rng.a*z", "locals": {
            "constants": {"scale": {"values": [1, 2]}},
            "rng": {"a": {"bank": "u", "transform": {"kind": "affine", "scale": "const.scale", "shift": 0}}}
        }}]}
        request["rhs"]["x1"] = "hole(R,x0)+hole(R,x1)"
        records, _ = collect(request)
        pool = records_of(records, "variant")[0]["pools"]
        self.assertEqual(len(pool["constants"]), 2)
        self.assertEqual(len(pool["rng_bindings"]), 2)
        self.assertEqual(pool["configuration_count"], 36)
        resolved = evaluate_prelude(pool, configuration_indices(pool, 35))
        self.assertEqual(list(resolved["const"].values()), [2, 2])

    def test_unused_axes_do_not_inflate_count(self):
        request = fixture()
        request["constants"] = {"unused": {"values": [1, 2, 3]}}
        request["leaves"] = {"unused": {"states": request["states"], "arity": 2}}
        _, summary = collect(request)
        self.assertEqual(summary["configuration_visits"], 1)

    def test_joint_sampling_caps_entire_group_product(self):
        request = fixture()
        request["states"] = ["x"+str(i) for i in range(5)]
        request["rhs"] = {state: "0" for state in request["states"]}
        request["rhs"]["x0"] = "leaf.a+leaf.b+leaf.c"
        request["leaves"] = {name: {"states": request["states"], "arity": 2} for name in ("a", "b", "c")}
        request["toggle_sampling"] = {"count": 7, "seed": 51}
        records, summary = collect(request)
        self.assertEqual(summary["unique_variants"], 7)
        self.assertEqual(summary["configuration_visits"], 56)
        again, _ = collect(request)
        self.assertEqual(records[:-1], again[:-1])

    def test_named_shape_choices_reuse_one_selection(self):
        request = fixture()
        request["shapes"] = {"force": {"choices": ["x0", "sin(x0)"]}}
        request["rhs"]["x1"] = "shape.force+shape.force"
        records, summary = collect(request)
        self.assertEqual(summary["unique_skeletons"], 2)
        for skeleton in records_of(records, "skeleton"):
            program = skeleton["rhs"]["x1"]
            midpoint = (len(program)-1)//2
            self.assertEqual(program[:midpoint], program[midpoint:-1])

    def test_compiled_rng_is_outside_rhs_loop(self):
        request = json.loads((ROOT / "examples/constant_rng_product.json").read_text())
        records, summary = collect(request)
        self.assertEqual(summary["configuration_visits"], 384)
        skeleton = records_of(records, "skeleton")[0]
        variant = records_of(records, "variant")[0]
        self.assertEqual(sum(i["op"] == "RNG_VALUE" for i in skeleton["rhs"]["x1"]), 3)
        setup = evaluate_prelude(variant["pools"], configuration_indices(variant["pools"], 383))
        kwargs = {"constants": setup["const"], "random_values": setup["rng"], "parameters": setup["param"]}
        result = evaluate_postorder(skeleton["rhs"]["x1"], [0.2, -0.3], **kwargs)
        expected = -setup["const"]["k"]*0.2+setup["rng"]["damping"]*0.3+setup["rng"]["force"]+setup["rng"]["offset"]
        self.assertAlmostEqual(result, expected)

    def test_limits_are_hard_and_whole_variant_is_atomic(self):
        request = fixture()
        request["leaves"] = {"a": {"states": request["states"], "arity": 2}}
        request["rhs"]["x1"] = "leaf.a"
        request["limits"] = {"max_configurations": 1}
        records, summary = collect(request)
        self.assertEqual(summary["unique_variants"], 0)
        self.assertEqual(summary["unique_skeletons"], 0)
        self.assertEqual(summary["stop_reason"], "max_configurations")
        self.assertFalse(records_of(records, "skeleton"))

    def test_structural_cap_finishes_accepted_toggle_variants(self):
        request = fixture()
        request["rules"] = {"R": ["leaf.a", "sin(leaf.a)"]}
        request["rhs"]["x1"] = "hole(R)"
        request["states"].append("x2")
        request["rhs"]["x2"] = "x1"
        request["leaves"] = {"a": {"states": request["states"], "arity": 2}}
        request["limits"] = {"max_skeletons": 1}
        _, summary = collect(request)
        self.assertEqual(summary["unique_skeletons"], 1)
        self.assertEqual(summary["unique_variants"], 3)
        self.assertEqual(summary["stop_reason"], "max_skeletons")

    def test_sqlite_and_memory_streams_match(self):
        request = fixture()
        request["rules"] = {"R": ["x0", "sin(x0)", "x0"]}
        request["rhs"]["x1"] = "hole(R)"
        memory, _ = collect(request)
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteDedup(Path(directory)/"dedup.sqlite")
            try:
                disk, _ = collect(request, dedup=store)
            finally:
                store.close()
            self.assertEqual(memory[:-1], disk[:-1])
            with self.assertRaises(CompileError):
                SQLiteDedup(Path(directory)/"dedup.sqlite")

    def test_integration_mode_never_falls_back(self):
        request = fixture()
        request["integration"] = {"method": "bdf", "stiff": True, "rtol": 1e-8}
        with self.assertRaises(CompileError):
            Compiler(request)
        records, _ = collect(request, annotation_only=True)
        info = records[0]["integration"]
        self.assertEqual(info["settings"]["method"], "bdf")
        self.assertFalse(info["backend_supported"])
        request["integration"] = {"method": "rk4", "stiff": True}
        with self.assertRaises(CompileError):
            Compiler(request)
        request["integration"] = {"method": "rk4", "rtol": 1e-8}
        with self.assertRaises(CompileError):
            Compiler(request)

    def test_unknown_branch_reference_fails_even_beyond_cap(self):
        request = fixture()
        request["rules"] = {"R": ["x0", "const.typo"]}
        request["rhs"]["x1"] = "hole(R)"
        request["limits"] = {"max_skeletons": 1}
        with self.assertRaises(CompileError):
            Compiler(request)

    def test_separate_lm_contract_counts_only_fitted_scalars(self):
        request = {"kind": "lm", "states": ["x"+str(i) for i in range(8)], "candidate_ids": ["candidate_1"], "parameters": ["p"+str(i) for i in range(8)]}
        result = compile_lm_request(request)
        self.assertEqual(result["fitted_dimension"], 8)
        self.assertFalse(result["optimizer_executed"])
        for key, addition in (("states", "x8"), ("parameters", "p8")):
            invalid = deepcopy(request)
            invalid[key].append(addition)
            with self.assertRaises(CompileError):
                compile_lm_request(invalid)
        invalid = deepcopy(request)
        invalid["parameters"] = ["p0", "p0"]
        with self.assertRaises(CompileError):
            compile_lm_request(invalid)

    def test_legacy_generate_request(self):
        request = fixture()
        rhs = request.pop("rhs")
        rhs["x1"] = "hole(R,x0)"
        request.update({"family_id": "legacy", "generate": {"rules": {"R(z)": ["z", "sin(z)"]}, "rhs": rhs, "expansion": "cartesian", "target_unique_skeletons": 10, "deduplicate": "lowered_system"}})
        _, summary = collect(request)
        self.assertEqual(summary["unique_skeletons"], 2)

    def test_rule_names_cannot_change_state_or_time_meaning(self):
        for name in ("x0", "t"):
            request = fixture()
            request["rules"] = {name: ["1"]}
            with self.assertRaises(CompileError):
                Compiler(request)
        request = fixture()
        request["rules"] = {"R(z)": ["z"]}
        request["rhs"]["x1"] = "R"
        with self.assertRaises(CompileError):
            Compiler(request)

    def test_invalid_legacy_containers_and_extreme_numbers_are_errors(self):
        for extra in (
            {"generate": {}, "expansion": []},
            {"budget": {"max_generation_attempts": 5}, "limits": []},
            {"version": 1.0},
            {"integration": {"method": "rk4", "dt": 10**400}},
        ):
            request = {**fixture(), **extra}
            with self.assertRaises(CompileError):
                Compiler(request)

    def test_cli_atomic_output_and_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"request.json", Path(directory)/"out.jsonl"
            source.write_text(json.dumps(fixture()))
            result = subprocess.run([sys.executable, "-m", "odegrammar", "compile", str(source), "-o", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = output.read_bytes()
            source.write_text('{"version":1,"version":2}')
            result = subprocess.run([sys.executable, "-m", "odegrammar", "compile", str(source), "-o", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("duplicate JSON key", result.stderr)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_compact_roundtrip_and_sqlite_error(self):
        from odegrammar.stream import expand_records
        result = subprocess.run([sys.executable, "-m", "odegrammar", "compile", "examples/constant_rng_product.json", "--compact"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        restored = list(expand_records(json.loads(line) for line in result.stdout.splitlines()))
        self.assertEqual(restored[-1]["configuration_visits"], 384)
        self.assertEqual(len(records_of(restored, "variant")), 1)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "odegrammar", "plan", "examples/oscillator.json", "--dedup-db", str(Path(directory)/"missing"/"dedup.sqlite")], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
