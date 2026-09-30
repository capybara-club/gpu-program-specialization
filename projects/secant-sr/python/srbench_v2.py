#!/usr/bin/env python3
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
"""Frozen constants for the published SRBench v2.0 protocols."""

from __future__ import annotations


SRBENCH_V2_PROTOCOL = "srbench-v2-groundtruth"
SRBENCH_V2_BLACKBOX_PROTOCOL = "srbench-v2-blackbox"
SRBENCH_V2_TRAIN_ROWS = 10_000
SRBENCH_V2_FEYNMAN_DATASET_COUNT = 116
SRBENCH_V2_FEYNMAN_MANIFEST_SHA256 = "754fb2a58a1bde27ef2f144f12aca7904b38746cfcc981626947f48a480e2988"
SRBENCH_V2_BLACKBOX_DATASET_COUNT = 122
SRBENCH_V2_BLACKBOX_MANIFEST_SHA256 = "af646147140f294128bc19be002f6c39016d3325466808ff056301547125cbed"
SRBENCH_V2_PMLB_REVISION = "v1.0.1.post3"
SRBENCH_V2_TARGET_NOISES = (0.0, 0.001, 0.01, 0.1)
SRBENCH_V2_SEEDS = (
    23654,
    15795,
    860,
    5390,
    16850,
    29910,
    4426,
    21962,
    14423,
    28020,
)
