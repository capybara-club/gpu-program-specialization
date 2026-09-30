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

import ast
import json
import sys


result = ast.literal_eval(sys.stdin.read())
samples = result["samples"]
summary = {
    "fit_thread_count": result["fit_thread_count"],
    "fit_count": result["fit_count"],
    "seconds": result["seconds"],
    "fits_per_second": result["fits_per_second"],
    "mean_iterations": result["mean_iterations"],
    "mean_accepted_steps": result["mean_accepted_steps"],
    "mean_factorization_attempts": result["mean_factorization_attempts"],
    "best_mse": result["best_mse"],
    "best_replay_mse": result["best_replay_mse"],
    "maximum_sample_replay_error": max(
        abs(sample["mse"] - sample["cpu_replay_mse"])
        for sample in samples
    ),
    "sample_0": samples[0],
    "sample_3": samples[min(3, len(samples) - 1)],
}
print(json.dumps(summary, sort_keys=True))
