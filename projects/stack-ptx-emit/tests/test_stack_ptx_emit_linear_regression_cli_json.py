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

import json
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_stack_ptx_emit_linear_regression_cli_json.py <cli-path>")

    completed = subprocess.run(
        [
            sys.argv[1],
            "--seed",
            "11",
            "--input-dim",
            "6",
            "--num-features",
            "5",
            "--steps",
            "9",
            "--eps",
            "1e-5",
            "--allowed-functions",
            "abs,neg,add,mul,fma",
            "--backends",
            "c,rust,numpy,latex,markdown",
            "--function-name",
            "demo_infer",
            "--input-names",
            "age,income,score,balance,exposure,target",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["request"]["seed"] == 11
    assert payload["request"]["input_dim"] == 6
    assert payload["request"]["num_features"] == 5
    assert payload["request"]["function_name"] == "demo_infer"
    assert abs(payload["request"]["eps"] - 1e-5) < 1e-8
    assert payload["request"]["allowed_functions"] == ["abs", "neg", "add", "mul", "fma"]
    assert payload["request"]["backends"] == ["c", "rust", "numpy", "latex", "markdown"]
    assert payload["regression"]["input_names"] == [
        "age",
        "income",
        "score",
        "balance",
        "exposure",
        "target",
    ]
    assert len(payload["regression"]["beta_unstandardized"]) == 5
    assert abs(
        payload["regression"]["beta_unstandardized"][0]
        - payload["regression"]["beta_standardized"][0] / payload["regression"]["stddevs"][0]
    ) < 1e-6
    assert len(payload["features"]) == 5
    assert set(payload["outputs"]) == {"c", "rust", "numpy", "latex", "markdown"}
    assert "demo_infer" in payload["outputs"]["c"]
    assert "pub fn demo_infer" in payload["outputs"]["rust"]
    assert "def demo_infer" in payload["outputs"]["numpy"]
    assert "\\documentclass" in payload["outputs"]["latex"]
    assert "# Linear Regression Model Specification" in payload["outputs"]["markdown"]
    assert payload["features"][0]["program"][-1] == "return"

    completed = subprocess.run(
        [
            sys.argv[1],
            "--seed",
            "5",
            "--input-dim",
            "4",
            "--num-features",
            "3",
            "--steps",
            "8",
            "--eps",
            "1e-4",
            "--allowed-functions",
            "sin,cos,log2,exp2,div,fma",
            "--backends",
            "markdown",
            "--function-name",
            "math_render_infer",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    math_payload = json.loads(completed.stdout)
    markdown_output = math_payload["outputs"]["markdown"]
    assert abs(math_payload["request"]["eps"] - 1e-4) < 1e-7
    assert math_payload["request"]["allowed_functions"] == ["sin", "cos", "log2", "exp2", "div", "fma"]
    assert "approx.ftz" not in markdown_output
    assert "approx.f32" not in markdown_output
    assert "\\sin" in markdown_output or "\\cos" in markdown_output or "\\log_2" in markdown_output or "\\frac{" in markdown_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
