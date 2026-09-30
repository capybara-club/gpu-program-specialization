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
from __future__ import annotations

import unittest

from secant_system_id.cli import _materialized_replay_verdict
from secant_system_id.fed_batch import INVALID_MSE


class MaterializedReplayVerdictTest(unittest.TestCase):
    def test_matching_finite_scores_pass(self) -> None:
        error, tolerance, passed = _materialized_replay_verdict(0.125, 0.12500001)
        self.assertLess(error, tolerance)
        self.assertTrue(passed)

    def test_false_low_score_fails(self) -> None:
        _error, _tolerance, passed = _materialized_replay_verdict(0.001, 4.0)
        self.assertFalse(passed)

    def test_matching_invalid_sentinels_fail(self) -> None:
        error, _tolerance, passed = _materialized_replay_verdict(INVALID_MSE, INVALID_MSE)
        self.assertEqual(error, 0.0)
        self.assertFalse(passed)

    def test_nonfinite_scores_fail(self) -> None:
        self.assertFalse(_materialized_replay_verdict(float("nan"), 1.0)[2])
        self.assertFalse(_materialized_replay_verdict(1.0, float("inf"))[2])


if __name__ == "__main__":
    unittest.main()
