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

import argparse
from importlib.metadata import version
from pathlib import Path
import statistics
import sys
import time

from common import positive_int, resume_records, write_csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from corpus import (
    DEFAULT_CORPUS,
    evaluate_numpy,
    load_corpus,
    lower_safe_operations,
    make_numpy_input,
)


SHAPES = ("materialize", "sse")
MAX_REPEATS = 256
CORPUS = None
INPUTS = 0
EXPRESSIONS = ()


def make_data(numpy, rows, seed):
    x = make_numpy_input(numpy, CORPUS, rows, seed)
    target = evaluate_numpy(
        numpy,
        EXPRESSIONS[CORPUS["target_expression"]],
        x)
    matrix = numpy.asfortranarray(numpy.column_stack((x, target)))
    return x, target, matrix


def evaluate_binary_node(numpy, operon, node_type):
    nodes = []
    for value in ((1.0, 2.0), (4.0, 3.0)):
        matrix = numpy.asfortranarray(
            numpy.array([[value[0], value[1], 0.0]], dtype=numpy.float32))
        dataset = operon.Dataset(matrix)
        left = operon.Node.Variable(1.0)
        left.HashValue = dataset.Variables[0].Hash
        right = operon.Node.Variable(1.0)
        right.HashValue = dataset.Variables[1].Hash
        tree = operon.Tree([left, right, operon.Node(node_type)]).UpdateNodes()
        output = float(operon.Evaluate(tree, dataset, operon.Range(0, 1))[0])
        nodes.append(output)
    return tuple(nodes)


def detect_min_max_node_types(numpy, operon):
    candidates = (operon.NodeType.Fmin, operon.NodeType.Fmax)
    observed = {
        candidate: evaluate_binary_node(numpy, operon, candidate)
        for candidate in candidates
    }
    minimum = next((node for node, values in observed.items() if values == (1.0, 3.0)), None)
    maximum = next((node for node, values in observed.items() if values == (2.0, 4.0)), None)
    if minimum is None or maximum is None or minimum == maximum:
        raise RuntimeError(f"could not identify PyOperon min/max semantics: {observed}")
    return minimum, maximum


def encode_expression(operon, operation_nodes, variables, expression, nodes):
    if isinstance(expression, int):
        node = operon.Node.Variable(1.0)
        node.HashValue = variables[expression].Hash
        nodes.append(node)
        return

    if expression[0] == "constant":
        nodes.append(operon.Node.Constant(float(expression[1])))
        return

    operation, *arguments = expression
    encoded_arguments = reversed(arguments) if len(arguments) == 2 else arguments
    for argument in encoded_arguments:
        encode_expression(
            operon,
            operation_nodes,
            variables,
            argument,
            nodes,
        )
    nodes.append(operation_nodes[operation]())


def make_trees(numpy, operon, dataset, asts):
    variables = dataset.Variables[:INPUTS]
    if len(variables) != INPUTS:
        raise RuntimeError("PyOperon dataset does not expose eight input variables")
    minimum_type, maximum_type = detect_min_max_node_types(numpy, operon)
    operation_nodes = {
        "add": operon.Node.Add,
        "sub": operon.Node.Sub,
        "mul": operon.Node.Mul,
        "div": operon.Node.Div,
        "min": lambda: operon.Node(minimum_type),
        "max": lambda: operon.Node(maximum_type),
        "abs": lambda: operon.Node(operon.NodeType.Abs),
        "sin": operon.Node.Sin,
        "cos": operon.Node.Cos,
        "sqrt": operon.Node.Sqrt,
        "tanh": operon.Node.Tanh,
    }

    fixed = []
    for expression in EXPRESSIONS[:min(asts, len(EXPRESSIONS))]:
        nodes = []
        encode_expression(
            operon,
            operation_nodes,
            variables,
            lower_safe_operations(expression),
            nodes,
        )
        tree = operon.Tree(nodes).UpdateNodes()
        if tree.Length != len(nodes):
            raise RuntimeError("the encoded Operon expression has an invalid length")
        fixed.append(tree)
    return [fixed[idx % len(fixed)] for idx in range(asts)]


def validate_trees(numpy, operon, trees, dataset, x):
    rows = min(257, x.shape[0])
    output = numpy.empty(len(trees) * rows, dtype=numpy.float32)
    operon.EvaluateTrees(
        trees,
        dataset,
        operon.Range(0, rows),
        output,
        1,
    )
    output = output.reshape(len(trees), rows)
    for ast_idx in range(len(trees)):
        expected = evaluate_numpy(
            numpy,
            EXPRESSIONS[ast_idx % len(EXPRESSIONS)],
            x[:rows],
        )
        maximum_error = float(numpy.max(
            numpy.abs(output[ast_idx] - expected) /
            (1.0 + numpy.abs(expected))))
        if maximum_error > 2.0e-4:
            raise RuntimeError(
                "fixed Operon expression validation failed: "
                f"ast={ast_idx} max_relative_error={maximum_error}")


def validate_sse(numpy, result, output, target):
    difference = output.astype(numpy.float64) - target[None, :]
    expected = numpy.sum(difference * difference, axis=1)
    if not numpy.allclose(result, expected, rtol=2.0e-5, atol=2.0e-4):
        error = numpy.abs(result - expected)
        worst_idx = int(numpy.argmax(error))
        raise RuntimeError(
            "fixed Operon SSE validation failed: "
            f"ast={worst_idx} actual={result[worst_idx]} "
            f"expected={expected[worst_idx]}")


def time_cpu(function, warmups, min_repeats, min_seconds):
    for _ in range(warmups):
        function()

    times = []
    total = 0.0
    while (
        (len(times) < min_repeats or total < min_seconds)
        and len(times) < MAX_REPEATS
    ):
        begin = time.perf_counter()
        function()
        seconds = time.perf_counter() - begin
        times.append(seconds)
        total += seconds

    if not times:
        raise RuntimeError("Operon timing produced no result")
    return min(times), statistics.median(times), len(times)


def collect_case(
    numpy,
    operon,
    trees,
    dataset,
    target,
    rows,
    workers,
    shape,
    warmups,
    min_repeats,
    min_seconds,
    seed,
):
    output = numpy.empty(len(trees) * rows, dtype=numpy.float32)
    output_2d = output.reshape(len(trees), rows)
    result = numpy.empty(len(trees), dtype=numpy.float32)
    scratch = numpy.empty_like(output_2d) if shape == "sse" else None
    data_range = operon.Range(0, rows)

    def materialize():
        operon.EvaluateTrees(
            trees,
            dataset,
            data_range,
            output,
            workers,
        )

    if shape == "materialize":
        function = materialize
        execution_mode = "evaluate_trees_preallocated"
        notes = f"profile={CORPUS['profile']};caller_owned_output"
    else:
        def function():
            materialize()
            numpy.subtract(output_2d, target[None, :], out=scratch)
            numpy.square(scratch, out=scratch)
            numpy.sum(scratch, axis=1, out=result)

        execution_mode = "evaluate_trees_then_numpy_sse"
        notes = (
            f"profile={CORPUS['profile']};"
            "caller_owned_output;not_fused_fitness")

    best, median, repeats = time_cpu(
        function,
        warmups,
        min_repeats,
        min_seconds,
    )
    if shape == "sse":
        validate_sse(numpy, result, output_2d, target)
    checksum = (
        float(output_2d[0, 0] + output_2d[-1, -1])
        if shape == "materialize"
        else float(result[0] + result[-1])
    )
    asts = len(trees)
    return {
        "system": "operon",
        "backend": "cpu_simd_runtime_dispatch",
        "shape": shape,
        "ast_mode": "alu",
        "corpus": CORPUS["name"],
        "corpus_hash": CORPUS["hash"],
        "seed": seed,
        "rows": rows,
        "asts": asts,
        "workers": workers,
        "execution_mode": execution_mode,
        "best_seconds": f"{best:.9f}",
        "median_seconds": f"{median:.9f}",
        "asts_per_second": f"{asts / median:.3f}",
        "row_evals_per_second": f"{asts * rows / median:.3f}",
        "repeats": repeats,
        "checksum": f"{checksum:.9g}",
        "notes": notes,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Collect Operon expression-backend runtime data.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--shapes", choices=SHAPES, nargs="+", default=list(SHAPES))
    parser.add_argument(
        "--rows",
        type=positive_int,
        nargs="+",
        default=[1_024, 16_384, 262_144],
    )
    parser.add_argument("--asts", type=positive_int, nargs="+", default=[256])
    parser.add_argument(
        "--workers",
        type=positive_int,
        nargs="+",
        default=[1, 12, 24],
    )
    parser.add_argument("--warmups", type=positive_int, default=2)
    parser.add_argument("--min-repeats", type=positive_int, default=5)
    parser.add_argument("--min-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=positive_int, default=1)
    args = parser.parse_args(argv)
    if args.min_seconds <= 0.0:
        parser.error("--min-seconds must be positive")
    for values, label in (
        (args.shapes, "shapes"),
        (args.rows, "rows"),
        (args.asts, "asts"),
        (args.workers, "workers"),
    ):
        if len(values) != len(set(values)):
            parser.error(f"--{label} contains duplicates")
    return args


def main(argv=None):
    global CORPUS, INPUTS, EXPRESSIONS

    args = parse_args(argv)
    CORPUS = load_corpus(args.corpus)
    INPUTS = CORPUS["num_inputs"]
    EXPRESSIONS = CORPUS["expressions"]

    import numpy as np
    import pyoperon as op

    key_fields = ("shape", "rows", "asts", "workers")
    records, completed = resume_records(
        args.output,
        key_fields,
        {
            "system": ("operon",),
            "backend": ("cpu_simd_runtime_dispatch",),
            "shape": args.shapes,
            "rows": args.rows,
            "asts": args.asts,
            "workers": args.workers,
            "seed": (args.seed,),
            "corpus": (CORPUS["name"],),
            "corpus_hash": (CORPUS["hash"],),
        },
    )
    for rows in args.rows:
        x, target, matrix = make_data(np, rows, args.seed)
        dataset = op.Dataset(matrix)
        for asts in args.asts:
            trees = make_trees(np, op, dataset, asts)
            validate_trees(np, op, trees, dataset, x)
            for shape in args.shapes:
                for workers in args.workers:
                    key = tuple(str(value) for value in (
                        shape,
                        rows,
                        asts,
                        workers,
                    ))
                    if key in completed:
                        print(
                            f"skip operon {shape} rows={rows} asts={asts} "
                            f"workers={workers}",
                            flush=True,
                        )
                        continue
                    record = collect_case(
                        np,
                        op,
                        trees,
                        dataset,
                        target,
                        rows,
                        workers,
                        shape,
                        args.warmups,
                        args.min_repeats,
                        args.min_seconds,
                        args.seed,
                    )
                    records.append(record)
                    completed.add(key)
                    write_csv(args.output, records)
                    print(
                        f"{record['system']} {shape} rows={rows} asts={asts} "
                        f"workers={workers} row_evals_per_second="
                        f"{float(record['row_evals_per_second']):.3e}",
                        flush=True,
                    )
    print(f"version pyoperon={version('pyoperon')}", flush=True)
    print(op.Version(), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
