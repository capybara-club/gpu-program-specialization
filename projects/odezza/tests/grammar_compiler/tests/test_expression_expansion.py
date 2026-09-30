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
import time
import unittest
from unittest.mock import patch

from odegrammar.expr import ExpressionError, Node, node_count, node_depth, parse_expr, rename_refs, substitute, walk
from odegrammar.expansion import GrammarError, GrammarExpander


class ExpressionTests(unittest.TestCase):
    def test_precedence_and_negative_integer_power(self):
        node = parse_expr("-x0 + leaf.a * sin(theta.k) / pow(rng.r, -2)")
        self.assertEqual(node.op, "add")
        self.assertEqual(node.children[0], Node("neg", children=(Node("symbol", "x0"),)))
        self.assertEqual(node.children[1].op, "div")
        self.assertEqual(node.children[1].children[1].value, -2)
        self.assertEqual(parse_expr("x0**2"), parse_expr("pow(x0,2)"))

    def test_reject_unsafe_or_ambiguous_syntax(self):
        for text in ("__import__('os').system('id')", "[x0]", "x0[0]", "True", "1e999", "x0^2", "pow(x0,x1)", "sin(x0,x1)", "leaf.a.b", "hole('R')"):
            with self.subTest(text=text), self.assertRaises(ExpressionError):
                parse_expr(text)

    def test_qualified_hygiene_and_tags(self):
        node = parse_expr("z + const.z")
        node = substitute(node, {"z": Node("symbol", "x0")})
        node = rename_refs(node, {("const", "z"): "bound"})
        wrapped = Node("tag", ("force",), (node,))
        self.assertEqual(node_count(wrapped), 3)
        self.assertEqual(node_depth(wrapped), 2)
        self.assertEqual(node.children[1], Node("const", "bound"))


class ExpansionTests(unittest.TestCase):
    def test_independent_holes_and_shared_shapes(self):
        expander = GrammarExpander({"R(z)": ["z", "sin(z)"]}, {"f": "hole(R,x0)"})
        independent = list(expander.expand({"x0": "hole(R,x0)+hole(R,x1)"}))
        self.assertEqual(len(independent), 4)
        shared = list(expander.expand({"x0": "shape.f+shape.f", "x1": "shape.f"}))
        self.assertEqual(len(shared), 2)
        for item in shared:
            self.assertEqual(item.rhs["x0"].children, (item.rhs["x1"], item.rhs["x1"]))

    def test_million_product_is_streamed(self):
        expander = GrammarExpander({"R(z)": ["z", "sin(z)", "cos(z)", "tanh(z)", "pow(z,2)", "z/(1+pow(z,2))", "sin(pow(z,2))", "cos(pow(z,2))", "tanh(pow(z,2))", "sin(tanh(z))"]}, {})
        first = list(itertools.islice(expander.expand({f"x{i}": f"hole(R,x{i})" for i in range(6)}), 3))
        self.assertEqual(len(first), 3)
        self.assertLess(expander.stats["steps"], 100)

    def test_recursive_bounded_enumeration(self):
        expander = GrammarExpander({"R(z)": ["z", "sin(R(z))"]}, {}, max_depth=3, max_expansion_depth=8)
        values = list(expander.expand({"x0": "R(x0)"}))
        self.assertEqual(len(values), 3)
        self.assertEqual([node_depth(x.rhs["x0"]) for x in values], [1, 2, 3])
        self.assertEqual(expander.stop_reason, "exhausted")

    def test_recursive_rule_without_terminal_stops(self):
        expander = GrammarExpander({"R": ["sin(R)"]}, {}, max_expansion_depth=5)
        self.assertEqual(list(expander.expand({"x0": "R"})), [])
        self.assertGreater(expander.stats["pruned_recursion"], 0)

    def test_local_bindings_fresh_per_hole_and_shape_shared(self):
        alternative = {"expr": "const.k*x0 + rng.r", "tags": ["growth"], "locals": {"constants": {"k": {"values": [1, 2]}}, "rng": {"r": {"bank": "u", "transform": {"kind": "affine", "scale": "const.k", "shift": 0}}}}}
        expander = GrammarExpander({"R": [alternative]}, {"f": "R"})
        independent = next(expander.expand({"x0": "R+R"}))
        self.assertEqual(len(independent.locals["constants"]), 2)
        self.assertEqual(len(independent.locals["rng"]), 2)
        refs = {n.value for n in walk(independent.rhs["x0"]) if n.op == "const"}
        self.assertEqual(refs, set(independent.locals["constants"]))
        for binding in independent.locals["rng"].values():
            self.assertIn(binding["transform"]["scale"][6:], refs)
        shared = next(expander.expand({"x0": "shape.f+shape.f"}))
        self.assertEqual(len(shared.locals["constants"]), 1)
        self.assertEqual(len([n for n in walk(shared.rhs["x0"]) if n.op == "tag"]), 2)

    def test_locals_do_not_capture_actual_arguments(self):
        expander = GrammarExpander({"R(z)": [{"expr": "z+const.k", "locals": {"constants": {"k": {"value": 2}}}}]}, {})
        item = next(expander.expand({"x0": "hole(R,const.k)"}))
        self.assertEqual(item.rhs["x0"].children[0], Node("const", "k"))
        self.assertTrue(item.rhs["x0"].children[1].value.startswith("__local_"))

    def test_actual_expression_selected_once(self):
        expander = GrammarExpander({"R": ["x0", "sin(x0)"], "Double(z)": ["z+z"]}, {})
        items = list(expander.expand({"x0": "Double(R)"}))
        self.assertEqual(len(items), 2)
        for item in items:
            a, b = item.rhs["x0"].children
            self.assertEqual(a, b)

    def test_seeded_sampling_and_step_limit(self):
        expander = GrammarExpander({"R": ["x0", "sin(x0)", "cos(x0)"]}, {})
        one = list(expander.expand({"x0": "R+R"}, strategy="sample", seed=42, max_derivations=10))
        two = list(expander.expand({"x0": "R+R"}, strategy="sample", seed=42, max_derivations=10))
        self.assertEqual(one, two)
        limited = GrammarExpander({"R": ["R+R", "x0"]}, {}, max_steps=20)
        list(limited.expand({"x0": "R"}))
        self.assertEqual(limited.stop_reason, "max_expansion_steps")
        self.assertEqual(limited.stats["steps"], 20)

    def test_unknown_rules_arity_and_shape_cycle(self):
        with self.assertRaises(GrammarError):
            GrammarExpander({"R": ["Missing(x0)"]}, {})
        with self.assertRaises(GrammarError):
            list(GrammarExpander({"R(z)": ["z"]}, {}).expand({"x0": "R"}))
        with self.assertRaises(GrammarError):
            list(GrammarExpander({"R(z)": ["z"]}, {}).expand({"x0": "R(x0,x1)"}))
        with self.assertRaises(GrammarError):
            list(GrammarExpander({}, {"a": "shape.b", "b": "shape.a"}).expand({"x0": "shape.a"}))

    def test_whole_system_node_budget(self):
        expander = GrammarExpander({}, {}, max_nodes=2)
        self.assertEqual(list(expander.expand({"x0": "sin(x0)", "x1": "x1"})), [])

    def test_already_expired_deadline(self):
        expander = GrammarExpander({}, {}, deadline=time.monotonic() - 1)
        self.assertEqual(list(expander.expand({"x0": "x0"})), [])
        self.assertEqual(expander.stop_reason, "max_seconds")
        self.assertEqual(expander.stats["steps"], 0)

    def test_deadline_interrupts_recursive_search_without_outputs(self):
        expander = GrammarExpander({"R": ["R+R", "sin(R)"]}, {}, deadline=10,
                                   max_expansion_depth=30, max_steps=10000)
        with patch("odegrammar.expansion.time.monotonic", side_effect=[0, 11]):
            self.assertEqual(list(expander.expand({"x0": "R"})), [])
        self.assertEqual(expander.stop_reason, "max_seconds")
        self.assertEqual(expander.stats["steps"], 128)

    def test_limit_status_does_not_claim_exhaustion_without_checking(self):
        expander = GrammarExpander({"R": ["x0", "sin(x0)"]}, {})
        self.assertEqual(len(list(expander.expand({"x0": "R"}, max_derivations=3))), 2)
        self.assertEqual(expander.stop_reason, "exhausted")
        self.assertEqual(len(list(expander.expand({"x0": "R"}, max_derivations=2))), 2)
        self.assertEqual(expander.stop_reason, "max_derivations")

    def test_alias_recursion_consumes_recursion_budget(self):
        expander = GrammarExpander({"A": ["B"], "B": ["A", "x0"]}, {}, max_expansion_depth=4)
        values = list(expander.expand({"x0": "A"}))
        self.assertEqual(len(values), 2)
        self.assertTrue(all(item.rhs["x0"] == Node("symbol", "x0") for item in values))
        self.assertGreater(expander.stats["pruned_recursion"], 0)


if __name__ == "__main__":
    unittest.main()
