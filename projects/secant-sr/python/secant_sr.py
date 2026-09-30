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
"""Standard-library client for the toggle-native C99 search executable."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile


OPTIONS = {
    "population", "generations", "seed", "seconds", "banks", "constants",
    "toggle_bits", "bank_seed", "distribution", "constant_min", "constant_max",
    "mean", "stddev", "max_nodes", "max_depth", "initial_depth", "elites",
    "operators", "stop_nmse", "toggle_probability", "four_way_probability",
    "coefficient_probability", "ast_batch", "score_mib", "pack", "kernels",
    "tile_rows", "threads", "workers", "streams", "device", "parsimony",
    "refine_rounds", "refine_budget", "refine_scale",
    "align_crossover_bits", "toggle_mutation_probability", "leaf_mix_probability",
    "refine_parameters", "power_mutation_probability",
    "finalists",
    "lm_iterations", "lm_budget", "lm_bindings", "lm_starts", "lm_parameters",
    "lm_interval", "lm_threads", "lm_scale",
}


def _matrix(X, y):
    X = [list(map(float, row)) for row in X]
    y = list(map(float, y))
    if len(X) != len(y) or len(X) < 2 or not X[0] or len(X[0]) > 128:
        raise ValueError("need matching X/y and at least two rows")
    if any(len(row) != len(X[0]) for row in X):
        raise ValueError("ragged input matrix")
    if any(not math.isfinite(v) for row in X for v in row) or any(
        not math.isfinite(v) for v in y
    ):
        raise ValueError("data must be finite")
    return X, y


def write_dataset(path, X, y, validation):
    """Write column-major little-endian f32 data, without formulas or seeds."""
    X, y = _matrix(X, y)
    V, w = _matrix(*validation)
    if len(V[0]) != len(X[0]):
        raise ValueError("validation input count differs")
    with open(path, "wb") as stream:
        stream.write(struct.pack("<8sIIQQ", b"SECSRDS\0", 1, len(X[0]), len(X), len(V)))
        for matrix, target in ((X, y), (V, w)):
            for column in zip(*matrix):
                stream.write(struct.pack("<" + "f" * len(column), *column))
            stream.write(struct.pack("<" + "f" * len(target), *target))


def fit(X, y, *, validation=None, executable=None, backend="cuda", **options):
    """Return fit metrics, replayable AST bytes, and generation progress.

    With no validation argument, reserve a deterministic 20% holdout (at least
    two rows), independent of the search seed. No validation data enters search.
    CUDA errors never trigger a CPU retry. The executable's time budget is
    cooperative between generations, not a hard subprocess deadline.
    """
    unknown = set(options) - OPTIONS
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    X, y = _matrix(X, y)
    if validation is None:
        if len(X) < 4:
            raise ValueError("need four rows to create train/validation splits")
        indices = list(range(len(X)))
        random.Random(314159).shuffle(indices)
        count = max(2, min(len(X) - 2, len(X) // 5))
        hold = set(indices[:count])
        validation = (
            [row for i, row in enumerate(X) if i in hold],
            [v for i, v in enumerate(y) if i in hold],
        )
        X, y = (
            [row for i, row in enumerate(X) if i not in hold],
            [v for i, v in enumerate(y) if i not in hold],
        )
    default = Path(__file__).resolve().parents[1] / "build-toggle/secant_sr_search"
    exe = Path(executable or default).resolve()
    with tempfile.TemporaryDirectory(prefix="secant-sr-") as directory:
        data = Path(directory) / "dataset.bin"
        write_dataset(data, X, y, validation)
        command = [str(exe), "--backend", backend, "--data", str(data)]
        for key, value in options.items():
            if key == "operators" and not isinstance(value, str):
                value = ",".join(value)
            command += ["--" + key.replace("_", "-"), str(value)]
        env = dict(os.environ)
        env.setdefault("CUDA_MODULE_LOADING", "EAGER")
        process = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        if process.returncode:
            raise RuntimeError(
                f"Secant-SR failed ({process.returncode}): {process.stderr.strip()}"
            )
        records = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
        if not records or records[-1].get("event") != "result":
            raise RuntimeError("search returned no final result")
        result = records[-1]
        result["progress"] = records[:-1]
        return result
