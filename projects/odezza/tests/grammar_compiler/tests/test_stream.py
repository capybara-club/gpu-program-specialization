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
import json
import unittest

from odegrammar.compiler import Compiler
from odegrammar.pools import PoolPlanner, configuration_indices, evaluate_prelude
from odegrammar.stream import StreamError, compact_records, expand_records


def variants_from(planner, active, skeleton="s"):
    for i, pools in enumerate(planner.iter_variants(active, skeleton)):
        yield {"type": "variant", "id": f"v{i}", "variant_id": f"v{i}", "skeleton_id": skeleton,
               "family_id": "mechanism", "tags": ["force"],
               "annotations": [{"start": 0, "end": 3, "tags": ["force"]}], "pools": pools}


def planner_fixture():
    states = [f"x{i}" for i in range(5)]
    planner = PoolPlanner(states,
        leaves={"a": {"states": states, "arity": 2}, "b": {"states": states, "arity": 4}},
        constant_banks={"rates": [1, 2, 4, 8]},
        constants={"k": {"bank": "rates"}, "j": {"bank": "rates"}, "fixed": {"value": 2}},
        rng_banks={"u": {"base": "uniform01", "count": 8, "scope": "run"},
                   "n": {"base": "normal01", "count": 8, "scope": "skeleton"}},
        rng={"u": {"bank": "u", "axis": "v", "transform": {"kind": "uniform", "low": 0, "high": "const.k"}},
             "n": {"bank": "n", "axis": "v", "transform": {"kind": "normal", "mean": 2, "std": "const.fixed"}}},
        parameters={"theta": {"initial": 3}})
    active = {"leaf": {"a", "b"}, "const": {"j"}, "rng": {"u", "n"}, "param": {"theta"}}
    return planner, active


class StreamTests(unittest.TestCase):
    def test_exact_roundtrip_preserves_all_metadata_and_original_ids(self):
        planner, active = planner_fixture()
        records = [
            {"type": "manifest", "format": "test", "states": planner.states},
            {"type": "skeleton", "id": "s", "rhs": {"x0": [{"op": "STATE", "index": 0}]}},
            *variants_from(planner, active),
            {"type": "provenance", "variant_id": "v0", "tags": ["alternate"]},
            {"type": "summary", "unique_variants": 50},
        ]
        original = copy.deepcopy(records)
        compact = list(compact_records(records))
        # Exercise the actual JSON wire format, not only Python object reuse.
        wire = [json.loads(json.dumps(record)) for record in compact]
        restored = list(expand_records(wire))
        self.assertEqual(restored, original)
        self.assertEqual(records, original)
        original_variant = records[2]
        restored_variant = restored[2]
        index = original_variant["pools"]["configuration_count"] - 1
        self.assertEqual(evaluate_prelude(original_variant["pools"], configuration_indices(original_variant["pools"], index)),
                         evaluate_prelude(restored_variant["pools"], configuration_indices(restored_variant["pools"], index)))

    def test_interns_constant_arrays_rng_descriptors_and_numeric_plan(self):
        planner, active = planner_fixture()
        compact = list(compact_records(variants_from(planner, active)))
        resources = [record for record in compact if record["type"] == "pool_resource"]
        self.assertEqual(sum(record["kind"] == "constant_bank" for record in resources), 1)
        self.assertEqual(sum(record["kind"] == "rng_bank" for record in resources), 2)
        self.assertEqual(sum(record["kind"] == "numeric_plan" for record in resources), 1)
        variants = [record for record in compact if record["type"] == "variant"]
        self.assertEqual(len(variants), 50)
        self.assertEqual(len({record["numeric_plan_id"] for record in variants}), 1)
        self.assertTrue(all("pools" not in record and "toggles" in record for record in variants))
        plan = next(record["data"] for record in resources if record["kind"] == "numeric_plan")
        self.assertNotIn("toggles", plan["pools"])
        self.assertNotIn("configuration_count", plan["pools"])
        self.assertTrue(all(axis["kind"] != "leaf" for axis in plan["pools"]["pool_axes"]))

    def test_empty_numeric_plan_and_different_toggle_arities_share_plan(self):
        first = next(variants_from(PoolPlanner(["x0"]), {}))
        second = next(variants_from(PoolPlanner(["x0", "x1"], leaves={"a": {"states": ["x0", "x1"], "arity": 2}}), {"leaf": {"a"}}))
        records = [first, second]
        compact = list(compact_records(records))
        plans = [record for record in compact if record.get("kind") == "numeric_plan"]
        self.assertEqual(len(plans), 1)
        restored = list(expand_records(compact))
        self.assertEqual(restored, records)
        self.assertEqual([r["pools"]["configuration_count"] for r in restored], [1, 2])

    def test_rng_bank_length_separate_from_stable_resource_identity(self):
        records = []
        for count in (4, 8):
            planner = PoolPlanner(["x0"], rng_banks={"u": {"base": "uniform01", "count": count}}, rng={"a": {"bank": "u"}})
            records.append(next(variants_from(planner, {"rng": {"a"}})))
        compact = list(compact_records(records))
        self.assertEqual(sum(r.get("kind") == "rng_bank" for r in compact), 1)
        self.assertEqual(sum(r.get("kind") == "numeric_plan" for r in compact), 2)
        self.assertEqual(list(expand_records(compact)), records)

    def test_bounded_seen_id_cache_can_reemit_resources(self):
        planner, active = planner_fixture()
        records = list(variants_from(planner, active))[:3]
        compact = list(compact_records(records, max_seen=1))
        ids = [r["id"] for r in compact if r["type"] == "pool_resource"]
        self.assertLess(len(set(ids)), len(ids))
        self.assertEqual(list(expand_records(compact)), records)
        self.assertEqual(list(expand_records(compact_records(records, max_seen=0))), records)

    def test_compaction_is_lazy_and_resources_precede_references(self):
        planner, active = planner_fixture()
        first = next(variants_from(planner, active))
        consumed = []

        def source():
            consumed.append(1)
            yield first
            consumed.append(2)
            raise AssertionError("eagerly read next input variant")

        iterator = compact_records(source())
        emitted = []
        while True:
            item = next(iterator)
            emitted.append(item)
            if item["type"] == "variant":
                break
        self.assertEqual(consumed, [1])
        self.assertEqual(list(expand_records(emitted)), [first])

    def test_large_bank_output_reduction(self):
        states = [f"x{i}" for i in range(10)]
        planner = PoolPlanner(states, leaves={"a": {"states": states}}, constants={"k": {"values": list(range(2000))}})
        records = list(variants_from(planner, {"leaf": {"a"}, "const": {"k"}}))
        compact = list(compact_records(records))
        raw_bytes = sum(len(json.dumps(record)) for record in records)
        compact_bytes = sum(len(json.dumps(record)) for record in compact)
        self.assertLess(compact_bytes, raw_bytes / 5)
        self.assertEqual(list(expand_records(compact)), records)

    def test_actual_compiler_stream_roundtrip_including_provenance(self):
        request = {"version": 1, "states": ["x0", "x1"], "integration": {"method": "rk4", "dt": 0.01},
                   "rhs": {"x0": "x1", "x1": "hole(R,x0) + const.k*leaf.a + rng.r"},
                   "rules": {"R(z)": [{"expr": "sin(z)", "tags": ["a"]}, {"expr": "sin(z)", "tags": ["b"]}]},
                   "leaves": {"a": {"states": ["x0", "x1"], "arity": 2}},
                   "constants": {"k": {"values": [1, 2, 3]}},
                   "rng_banks": {"u": {"base": "uniform01", "count": 3}}, "rng": {"r": {"bank": "u"}}}
        records = list(Compiler(request).records())
        self.assertTrue(any(r["type"] == "provenance" for r in records))
        self.assertEqual(list(expand_records(compact_records(records))), records)

    def test_reader_outputs_do_not_mutate_resource_cache(self):
        records = list(variants_from(PoolPlanner(["x0"], constants={"a": {"values": [1, 2]}}), {"const": {"a"}})) * 2
        reader = expand_records(compact_records(records))
        first = next(reader)
        first["pools"]["constants"]["a"]["values"][0] = 999
        second = next(reader)
        self.assertEqual(second["pools"]["constants"]["a"]["values"], [1, 2])

    def test_non_variant_records_and_raw_variants_pass_through(self):
        records = [{"type": "lm_request", "parameters": []}, {"type": "error", "message": "example"}]
        self.assertEqual(list(expand_records(compact_records(records))), records)
        raw = next(variants_from(PoolPlanner(["x0"]), {}))
        self.assertEqual(list(expand_records([raw])), [raw])

    def test_noncanonical_leaf_metadata_and_bad_count_rejected(self):
        planner, active = planner_fixture()
        original = next(variants_from(planner, active))
        bad = copy.deepcopy(original)
        bad["pools"]["pool_axes"][0]["extra"] = "cannot lose this"
        with self.assertRaisesRegex(StreamError, "noncanonical"):
            list(compact_records([bad]))
        bad = copy.deepcopy(original)
        bad["pools"]["configuration_count"] += 1
        with self.assertRaisesRegex(StreamError, "configuration_count"):
            list(compact_records([bad]))

    def test_missing_tampered_and_wrong_format_resources_rejected(self):
        planner, active = planner_fixture()
        compact = list(compact_records([next(variants_from(planner, active))]))
        with self.assertRaisesRegex(StreamError, "missing"):
            list(expand_records([compact[-1]]))
        bad = copy.deepcopy(compact)
        bad[0]["data"].append(999)
        with self.assertRaisesRegex(StreamError, "content"):
            list(expand_records(bad))
        bad = copy.deepcopy(compact)
        bad[0]["format"] = "future"
        with self.assertRaisesRegex(StreamError, "format"):
            list(expand_records(bad))

    def test_reject_already_compact_input_and_bad_cache_size(self):
        variant = next(variants_from(PoolPlanner(["x0"]), {}))
        compact = list(compact_records([variant]))
        with self.assertRaises(StreamError):
            list(compact_records(compact))
        for size in (-1, True, 1.5):
            with self.assertRaises(StreamError):
                list(compact_records([variant], max_seen=size))


if __name__ == "__main__":
    unittest.main()
