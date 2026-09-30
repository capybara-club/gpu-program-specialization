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
import itertools
import math
import unittest

from odegrammar.pools import (PoolError, PoolPlanner, _sample_ranks,
                              _unrank_combination, bank_value,
                              configuration_indices, evaluate_prelude,
                              materialize_bank)


STATES = [f"x{i}" for i in range(5)]


class ToggleTests(unittest.TestCase):
    def test_binary_exhaustive_pairs_and_visits(self):
        planner = PoolPlanner(STATES, leaves={"a": {"states": list(reversed(STATES)), "arity": 2}})
        variants = list(planner.iter_variants({"leaf": {"a"}}, "s"))
        self.assertEqual([tuple(v["toggles"]["a"]) for v in variants], list(itertools.combinations(range(5), 2)))
        self.assertEqual(planner.variant_count({"leaf": {"a"}}), 10)
        self.assertEqual(sum(v["configuration_count"] for v in variants), 20)
        covered = {evaluate_prelude(v, {"leaf:a": i})["leaf"]["a"] for v in variants for i in range(2)}
        self.assertEqual(covered, set(range(5)))

    def test_quad_exhaustive_groups(self):
        planner = PoolPlanner(STATES, leaves={"q": {"states": STATES, "arity": 4}})
        variants = list(planner.iter_variants({"leaf": {"q"}}, "s"))
        self.assertEqual(len(variants), 5)
        self.assertTrue(all(len(v["toggles"]["q"]) == 4 and v["configuration_count"] == 4 for v in variants))

    def test_independent_groups_cartesian_and_only_active_axes(self):
        planner = PoolPlanner(STATES, leaves={
            "a": {"states": STATES[:3], "arity": 2},
            "b": {"states": STATES[2:], "arity": 2},
            "unused": {"states": STATES, "arity": 4},
        })
        variants = list(planner.iter_variants({"leaf": {"a", "b"}}, "s"))
        self.assertEqual(len(variants), 9)
        self.assertTrue(all(v["configuration_count"] == 4 for v in variants))
        self.assertTrue(all(set(v["toggles"]) == {"a", "b"} for v in variants))

    def test_sample_unique_reproducible_and_independent_of_skeleton_order(self):
        planner = PoolPlanner(STATES, leaves={"a": {"states": STATES, "arity": 2, "coverage": "sample", "samples": 4, "seed": 8}})
        first = list(planner.iter_variants({"leaf": {"a"}}, "first"))
        second = list(planner.iter_variants({"leaf": {"a"}}, "second"))
        groups = [tuple(v["toggles"]["a"]) for v in first]
        self.assertEqual(len(set(groups)), 4)
        self.assertEqual(first, second)

    def test_oversample_caps_to_exhaustive(self):
        planner = PoolPlanner(STATES, leaves={"a": {"states": STATES, "arity": 4, "coverage": "sample", "samples": 999}})
        variants = list(planner.iter_variants({"leaf": {"a"}}, "s"))
        self.assertEqual(len(variants), 5)

    def test_explicit_groups_normalized(self):
        planner = PoolPlanner(STATES, leaves={"a": {"states": STATES, "arity": 2, "coverage": "explicit", "groups": [["x3", "x0"], ["x4", "x1"]]}})
        variants = list(planner.iter_variants({"leaf": {"a"}}, "s"))
        self.assertEqual([v["toggles"]["a"] for v in variants], [[0, 3], [1, 4]])

    def test_combination_unranking_matches_itertools(self):
        for n in range(2, 15):
            for k in (2, 4):
                if k <= n:
                    expected = list(itertools.combinations(range(n), k))
                    actual = [_unrank_combination(n, k, i) for i in range(len(expected))]
                    self.assertEqual(expected, actual)

    def test_rank_sampling_huge_domain_is_lazy(self):
        ranks = list(_sample_ranks(10**30, 10, 42))
        self.assertEqual(len(set(ranks)), 10)
        self.assertTrue(all(0 <= r < 10**30 for r in ranks))
        self.assertEqual(len(set(_sample_ranks(100, 100, 42))), 100)

    def test_joint_group_sampling_is_n_total_not_n_per_leaf(self):
        planner = PoolPlanner(STATES, leaves={name: {"states": STATES, "arity": 2} for name in ("a", "b", "c")})
        active = {"leaf": {"a", "b", "c"}}
        self.assertEqual(planner.variant_count(active), 1000)
        variants = list(planner.iter_variants(active, "s", group_sampling={"count": 7, "seed": 23}))
        self.assertEqual(len(variants), 7)
        keys = [tuple(tuple(v["toggles"][name]) for name in ("a", "b", "c")) for v in variants]
        self.assertEqual(len(set(keys)), 7)
        self.assertTrue(all(v["configuration_count"] == 8 for v in variants))
        self.assertEqual(variants, list(planner.iter_variants({"leaf": ["c", "b", "a"]}, "different", group_sampling={"count": 7, "seed": 23})))

    def test_joint_group_sampling_respects_explicit_and_sampled_leaf_domains(self):
        planner = PoolPlanner(STATES, leaves={
            "a": {"states": STATES, "coverage": "sample", "samples": 4, "seed": 17},
            "b": {"states": STATES, "coverage": "explicit", "groups": [["x0", "x3"], ["x1", "x4"]]},
            "c": {"states": STATES, "arity": 4},
        })
        active = {"leaf": {"a", "b", "c"}}
        exhaustive = list(planner.iter_variants(active, "s"))
        self.assertEqual(len(exhaustive), 40)
        sampled = list(planner.iter_variants(active, "s", group_sampling={"count": 7, "seed": 1}))
        self.assertEqual(len(sampled), 7)
        self.assertTrue(all(v in exhaustive for v in sampled))
        self.assertEqual(exhaustive, list(planner.iter_variants(active, "s", group_sampling={"count": 100, "seed": 1})))

    def test_joint_group_sampling_huge_product_without_expansion(self):
        names = [f"a{i}" for i in range(30)]
        planner = PoolPlanner(STATES, leaves={name: {"states": STATES} for name in names})
        active = {"leaf": set(names)}
        self.assertEqual(planner.variant_count(active), 10**30)
        variants = list(planner.iter_variants(active, "s", group_sampling={"count": 3}))
        self.assertEqual(len(variants), 3)
        self.assertTrue(all(len(v["toggles"]) == 30 for v in variants))

    def test_joint_sampling_validation_and_no_active_leaves(self):
        planner = PoolPlanner(STATES)
        self.assertEqual(len(list(planner.iter_variants({}, "s", group_sampling={"count": 7}))), 1)
        for spec in ({}, {"count": 0}, {"count": True}, {"count": 3, "seed": "bad"}, {"count": 3, "unknown": 2}):
            with self.subTest(spec=spec), self.assertRaises(PoolError):
                list(planner.iter_variants({}, "s", group_sampling=spec))

    def test_reject_invalid_toggle_declarations_even_unused(self):
        invalid = [
            {"states": STATES, "arity": 3},
            {"states": STATES, "arity": True},
            {"states": STATES[:3], "arity": 4},
            {"states": ["x0", "x0"], "arity": 2},
            {"states": ["x0", "bad"], "arity": 2},
            {"states": STATES, "coverage": "sample"},
            {"states": STATES, "coverage": "sample", "samples": 0},
            {"states": STATES, "coverage": "explicit", "groups": [["x0", "x0"]]},
            {"states": STATES, "coverage": "explicit", "groups": [["x0", "x1"], ["x1", "x0"]]},
            {"states": STATES, "groups": [["x0", "x1"]]},
        ]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(PoolError):
                PoolPlanner(STATES, leaves={"unused": spec})


class NumericPoolTests(unittest.TestCase):
    def test_constant_bank_cartesian_and_fixed_constants(self):
        planner = PoolPlanner(STATES, constants={"a": {"bank": "rates"}, "b": {"values": [10, 20, 30]}, "c": {"value": 7}}, constant_banks={"rates": [1, 2]})
        variant = next(planner.iter_variants({"const": {"a", "b", "c"}}, "s"))
        self.assertEqual(variant["configuration_count"], 6)
        values = {tuple(evaluate_prelude(variant, {"const:a": a, "const:b": b})["const"].values())
                  for a in range(2) for b in range(3)}
        self.assertEqual(len(values), 6)
        self.assertEqual(variant["parameter_initials"], {})

    def test_separate_slots_using_same_bank_remain_independent(self):
        planner = PoolPlanner(STATES, constants={"a": {"bank": "b"}, "c": {"bank": "b"}}, constant_banks={"b": [1, 2, 3]})
        variant = next(planner.iter_variants({"const": {"a", "c"}}, "s"))
        self.assertEqual(variant["configuration_count"], 9)

    def test_shared_draw_axis_zips_but_distinct_streams_are_independent(self):
        planner = PoolPlanner(STATES, rng_banks={"u": {"base": "uniform01", "count": 32}}, rng={"a": {"bank": "u", "axis": "vector"}, "b": {"bank": "u", "axis": "vector"}})
        variant = next(planner.iter_variants({"rng": {"a", "b"}}, "s"))
        self.assertEqual(variant["configuration_count"], 32)
        self.assertEqual(len(variant["rng_bank_requests"]), 2)
        values = evaluate_prelude(variant)["rng"]
        self.assertNotEqual(values["a"], values["b"])

    def test_default_draw_axes_cartesian_and_explicit_axis_cannot_collide(self):
        planner = PoolPlanner(STATES, rng_banks={"u": {"base": "uniform01", "count": 4}}, rng={"a": {"bank": "u"}, "b": {"bank": "u", "axis": "a"}})
        variant = next(planner.iter_variants({"rng": {"a", "b"}}, "s"))
        self.assertEqual(variant["configuration_count"], 16)

    def test_explicit_same_stream_shares_bank_values(self):
        planner = PoolPlanner(STATES, rng_banks={"u": {"base": "uniform01", "count": 8}}, rng={"a": {"bank": "u", "axis": "v", "stream": "shared"}, "b": {"bank": "u", "axis": "v", "stream": "shared"}})
        variant = next(planner.iter_variants({"rng": {"a", "b"}}, "s"))
        self.assertEqual(len(variant["rng_bank_requests"]), 1)
        for i in range(8):
            values = evaluate_prelude(variant, {"rng:shared:v": i})["rng"]
            self.assertEqual(values["a"], values["b"])

    def test_run_vs_skeleton_scopes_and_counter_stability(self):
        for scope in ("run", "skeleton"):
            planner = PoolPlanner(STATES, rng_banks={"u": {"base": "uniform01", "count": 10, "scope": scope, "seed": 19}}, rng={"a": {"bank": "u"}})
            variants = [next(planner.iter_variants({"rng": {"a"}}, s)) for s in ("s1", "s2", "s1")]
            requests = [v["rng_bank_requests"][0] for v in variants]
            self.assertEqual(requests[0], requests[2])
            self.assertEqual(requests[0] == requests[1], scope == "run")
            samples = materialize_bank(requests[0])
            self.assertEqual(samples, [bank_value(requests[0], i) for i in range(10)])
            self.assertEqual(list(reversed(samples)), [bank_value(requests[0], i) for i in reversed(range(10))])
            extended = dict(requests[0], count=100)
            self.assertEqual(samples, [bank_value(extended, i) for i in range(10)])

    def test_transform_dependencies_activate_constant_axes(self):
        planner = PoolPlanner(STATES, constants={"scale": {"values": [2, 4]}, "shift": {"value": 5}, "unused": {"values": [1, 2, 3]}}, rng_banks={"u": {"base": "uniform01", "count": 3}}, rng={"a": {"bank": "u", "transform": {"kind": "affine", "scale": "const.scale", "shift": "const.shift"}}}, parameters={"k": {"initial": 2}})
        variant = next(planner.iter_variants({"rng": {"a"}, "param": {"k"}}, "s"))
        self.assertEqual(variant["configuration_count"], 6)
        self.assertEqual(set(variant["constants"]), {"scale", "shift"})
        self.assertEqual(variant["parameter_initials"], {"k": 2})
        raw = bank_value(variant["rng_bank_requests"][0], 0)
        self.assertEqual(evaluate_prelude(variant)["rng"]["a"], 2 * raw + 5)
        self.assertEqual(evaluate_prelude(variant, {"const:scale": 1})["rng"]["a"], 4 * raw + 5)
        self.assertEqual(variant["prelude"][0]["frequency"], "once_per_configuration")

    def test_transform_distributions_and_base_validation(self):
        configs = [
            ("uniform01", {"kind": "uniform", "low": 2, "high": 3}),
            ("uniform01", {"kind": "log_uniform", "low": 2, "high": 100}),
            ("normal01", {"kind": "normal", "mean": 2, "std": 3}),
        ]
        for base, transform in configs:
            planner = PoolPlanner(STATES, rng_banks={"b": {"base": base, "count": 100}}, rng={"a": {"bank": "b", "transform": transform}})
            variant = next(planner.iter_variants({"rng": {"a"}}, "s"))
            values = [evaluate_prelude(variant, {"rng:slot:a": i})["rng"]["a"] for i in range(100)]
            self.assertTrue(all(math.isfinite(v) for v in values))
            if "low" in transform:
                self.assertTrue(all(transform["low"] <= v <= transform["high"] for v in values))

    def test_invalid_numeric_declarations(self):
        invalid = [
            {"constants": {"x": {"values": []}}},
            {"constant_banks": {"b": [float("nan")]}},
            {"constants": {"x": {"value": True}}},
            {"constants": {"x": {"bank": "missing"}}},
            {"constants": {"x": {"value": 1, "values": [2]}}},
            {"rng_banks": {"b": {"base": "normal01", "count": 0}}},
            {"rng_banks": {"b": {"base": "bad", "count": 1}}},
            {"parameters": {"k": {"initial": float("inf")}}},
            {"rng": {"x": {"bank": "missing"}}},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(PoolError):
                PoolPlanner(STATES, **kwargs)

    def test_invalid_transforms_even_unused(self):
        invalid = [
            {"kind": "normal"},
            {"kind": "uniform", "low": 2, "high": 1},
            {"kind": "log_uniform", "low": -1, "high": 3},
            {"kind": "uniform", "low": "const.low", "high": 4},
            {"kind": "affine", "scale": "const.missing"},
            {"kind": "affine", "scale": "theta.a"},
            {"kind": "uniform", "low": 0, "high": 1, "ignored": 7},
        ]
        for transform in invalid:
            with self.subTest(transform=transform), self.assertRaises(PoolError):
                PoolPlanner(STATES, constants={"low": {"values": [1, 8]}}, rng_banks={"u": {"base": "uniform01", "count": 2}}, rng={"a": {"bank": "u", "transform": transform}})

    def test_shared_axis_count_mismatch_rejected(self):
        with self.assertRaisesRegex(PoolError, "equal bank counts"):
            PoolPlanner(STATES, rng_banks={"a": {"base": "uniform01", "count": 2}, "b": {"base": "normal01", "count": 3}}, rng={"x": {"bank": "a", "axis": "v"}, "y": {"bank": "b", "axis": "v"}})

    def test_no_active_pools_is_one_configuration(self):
        variant = next(PoolPlanner(STATES).iter_variants({}, "s"))
        self.assertEqual(variant["configuration_count"], 1)
        self.assertEqual(variant["pool_axes"], [])
        self.assertEqual(configuration_indices(variant, 0), {})

    def test_flat_index_decoding_matches_cartesian_product(self):
        planner = PoolPlanner(STATES, leaves={"a": {"states": STATES[:2]}}, constants={"b": {"values": [5, 6, 7]}}, rng_banks={"u": {"base": "uniform01", "count": 4}}, rng={"c": {"bank": "u"}})
        variant = next(planner.iter_variants({"leaf": {"a"}, "const": {"b"}, "rng": {"c"}}, "s"))
        expected = list(itertools.product(range(2), range(3), range(4)))
        actual = [tuple(configuration_indices(variant, i).values()) for i in range(24)]
        self.assertEqual(actual, expected)
        for bad in (-1, 24, True):
            with self.assertRaises(PoolError):
                configuration_indices(variant, bad)

    def test_flat_index_for_enormous_implicit_product(self):
        huge = 10**30
        planner = PoolPlanner(STATES, rng_banks={"u": {"base": "uniform01", "count": huge}}, rng={"a": {"bank": "u"}, "b": {"bank": "u"}})
        variant = next(planner.iter_variants({"rng": {"a", "b"}}, "s"))
        self.assertEqual(variant["configuration_count"], huge * huge)
        self.assertEqual(configuration_indices(variant, huge * huge - 1), {"rng:slot:a": huge - 1, "rng:slot:b": huge - 1})
        values = evaluate_prelude(variant, configuration_indices(variant, huge * huge - 1))
        self.assertTrue(all(0 <= x < 1 for x in values["rng"].values()))

    def test_numeric_transform_overflow_is_an_explicit_failure(self):
        planner = PoolPlanner(STATES, rng_banks={"n": {"base": "normal01", "count": 100}}, rng={"a": {"bank": "n", "transform": {"kind": "affine", "scale": 1.7e308, "shift": 1.7e308}}})
        variant = next(planner.iter_variants({"rng": {"a"}}, "s"))
        failed = 0
        for i in range(100):
            try:
                values = evaluate_prelude(variant, {"rng:slot:a": i})
                self.assertTrue(math.isfinite(values["rng"]["a"]))
            except PoolError:
                failed += 1
        self.assertGreater(failed, 0)

    def test_large_uniform_bounds_do_not_overflow_difference(self):
        planner = PoolPlanner(STATES, rng_banks={"u": {"base": "uniform01", "count": 100}}, rng={"a": {"bank": "u", "transform": {"kind": "uniform", "low": -1.7e308, "high": 1.7e308}}})
        variant = next(planner.iter_variants({"rng": {"a"}}, "s"))
        self.assertTrue(all(math.isfinite(evaluate_prelude(variant, {"rng:slot:a": i})["rng"]["a"]) for i in range(100)))

    def test_revalidation_removes_deleted_declarations(self):
        planner = PoolPlanner(STATES, constants={"a": {"value": 1}})
        del planner.constants["a"]
        planner.validate()
        with self.assertRaises(PoolError):
            planner.resolve_active({"const": {"a"}})

    def test_reference_bounds_and_unknown_slots(self):
        planner = PoolPlanner(STATES, rng_banks={"u": {"base": "uniform01", "count": 4}}, rng={"a": {"bank": "u"}})
        variant = next(planner.iter_variants({"rng": {"a"}}, "s"))
        request = variant["rng_bank_requests"][0]
        for index in (-1, 4, True):
            with self.assertRaises(PoolError):
                bank_value(request, index)
        with self.assertRaises(PoolError):
            materialize_bank(request, max_count=3)
        with self.assertRaises(PoolError):
            evaluate_prelude(variant, {"rng:slot:a": 4})
        with self.assertRaises(PoolError):
            evaluate_prelude(variant, {"missing": 0})
        with self.assertRaises(PoolError):
            planner.resolve_active({"rng": {"missing"}})


if __name__ == "__main__":
    unittest.main()
