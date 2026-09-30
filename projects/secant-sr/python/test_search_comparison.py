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
import unittest
from compare_search_policies import settings_result


class ComparisonAudit(unittest.TestCase):
    def setUp(self):
        self.log = (
            "problem=external generation=2 train_r2=1 validation_r2=nan "
            "materialize_validation_diverged=0 elapsed_seconds=3.2 "
            "stop_reached=1 time_limit_reached=0\n"
            "final_cpu_optimizer enabled=0 train_r2_after=0.9999999 "
            "validation_r2_after=0.9998\n"
            "problem=external best_expression=(x0 + x1)\n")

    def test_final_cpu_metrics_are_authoritative(self):
        r = settings_result(self.log, "external")
        self.assertEqual(r["validation_r2"], .9998)
        self.assertEqual(r["accuracy_solution"], 1)
        self.assertEqual(r["generations"], 3)
        self.assertEqual(r["best_expression"], "(x0 + x1)")

    def test_fail_closed(self):
        for bad in (self.log.replace("diverged=0", "diverged=1"),
                    self.log.replace("enabled=0", "enabled=1"),
                    self.log.replace("validation_r2_after=0.9998", "validation_r2_after=nan"),
                    self.log.split("final_cpu_optimizer")[0]):
            with self.assertRaises(ValueError):
                settings_result(bad, "external")


if __name__ == "__main__":
    unittest.main()
