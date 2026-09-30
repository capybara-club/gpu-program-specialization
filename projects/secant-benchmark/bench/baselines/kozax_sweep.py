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
    make_numpy_input,
    postorder,
)


SHAPES = ("materialize", "sse")
MAX_REPEATS = 256
CORPUS = None
INPUTS = 0
EXPRESSIONS = ()
MAX_NODES = 0


def make_data(numpy, rows, seed):
    x = make_numpy_input(numpy, CORPUS, rows, seed)
    target = evaluate_numpy(
        numpy,
        EXPRESSIONS[CORPUS["target_expression"]],
        x)
    return x, target


def make_strategy(jax_numpy, genetic_programming, fitness_type, asts):
    if asts % 2 != 0:
        raise RuntimeError("Kozax requires an even population size")
    functions = {
        "add": (jax_numpy.add, 2),
        "sub": (jax_numpy.subtract, 2),
        "mul": (jax_numpy.multiply, 2),
        "div": (jax_numpy.divide, 2),
        "min": (jax_numpy.minimum, 2),
        "max": (jax_numpy.maximum, 2),
        "safe_div": (
            lambda x, y: x * y / (y * y + jax_numpy.float32(0.01)),
            2,
        ),
        "neg": (jax_numpy.negative, 1),
        "abs": (jax_numpy.abs, 1),
        "sin": (jax_numpy.sin, 1),
        "cos": (jax_numpy.cos, 1),
        "sqrt": (jax_numpy.sqrt, 1),
        "tanh": (jax_numpy.tanh, 1),
        "safe_sqrt": (
            lambda x: jax_numpy.sqrt(
                jax_numpy.abs(x) + jax_numpy.float32(0.01)),
            1,
        ),
        "safe_rsqrt": (
            lambda x: jax_numpy.reciprocal(jax_numpy.sqrt(
                jax_numpy.abs(x) + jax_numpy.float32(0.01))),
            1,
        ),
    }
    operators = [
        (name, functions[name][0], functions[name][1], 1.0)
        for name in CORPUS["operations"]
    ]
    return genetic_programming(
        1,
        asts,
        fitness_type(),
        operator_list=operators,
        variable_list=[[f"x{idx}" for idx in range(INPUTS)]],
        max_init_depth=8,
        max_nodes=MAX_NODES,
        device_type="gpu",
        constant_sd=0.0,
    )


def encode_expression(strategy, expression, nodes):
    if isinstance(expression, int):
        nodes.append([
            strategy.string_to_node[f"x{expression}"],
            -1,
            -1,
            0.0,
        ])
        return len(nodes) - 1

    if expression[0] == "constant":
        nodes.append([1, -1, -1, float(expression[1])])
        return len(nodes) - 1

    operation, *arguments = expression
    left_idx = encode_expression(strategy, arguments[0], nodes)
    right_idx = (
        encode_expression(strategy, arguments[1], nodes)
        if len(arguments) == 2
        else -1
    )
    nodes.append([
        strategy.string_to_node[operation],
        left_idx,
        right_idx,
        0.0,
    ])
    return len(nodes) - 1


def make_population(numpy, jax_numpy, strategy, asts):
    encoded = []
    for expression in EXPRESSIONS[:min(asts, len(EXPRESSIONS))]:
        nodes = []
        root = encode_expression(strategy, expression, nodes)
        if len(nodes) > MAX_NODES or root != len(nodes) - 1:
            raise RuntimeError("the encoded Kozax expression is invalid")
        padding = MAX_NODES - len(nodes)
        for node in nodes:
            if node[1] >= 0:
                node[1] += padding
            if node[2] >= 0:
                node[2] += padding
        padded = [[0, -1, -1, 0.0] for _ in range(padding)] + nodes
        encoded.append(numpy.asarray(padded, dtype=numpy.float32)[None, :, :])
    population = numpy.stack(
        [encoded[idx % len(encoded)] for idx in range(asts)],
        axis=0,
    )
    return jax_numpy.asarray(population)


def estimate_working_bytes(rows, asts):
    tree_state_bytes = rows * asts * MAX_NODES * 4 * 4
    return tree_state_bytes * 4


def make_materialize_function(jax, strategy):
    evaluate_rows = jax.vmap(strategy.tree_evaluator, in_axes=(None, 0))
    evaluate_population = jax.vmap(evaluate_rows, in_axes=(0, None))
    return jax.jit(evaluate_population)


def validate_materialize(numpy, output, x):
    actual = numpy.asarray(output)
    if actual.ndim != 3 or actual.shape[2] != 1:
        raise RuntimeError(f"unexpected Kozax materialize shape: {actual.shape}")
    check_rows = min(257, x.shape[0])
    expected_cases = [
        evaluate_numpy(numpy, expression, x[:check_rows])
        for expression in EXPRESSIONS[:min(actual.shape[0], len(EXPRESSIONS))]
    ]
    for ast_idx in range(actual.shape[0]):
        expected = expected_cases[ast_idx % len(expected_cases)]
        maximum_error = float(
            numpy.max(numpy.abs(actual[ast_idx, :check_rows, 0] - expected)))
        if maximum_error > 2.0e-5:
            raise RuntimeError(
                "fixed Kozax materialize validation failed: "
                f"ast={ast_idx} max_abs_error={maximum_error}")


def validate_fitness(numpy, output, x, target):
    actual = numpy.asarray(output)
    expected_cases = []
    for expression in EXPRESSIONS:
        prediction = evaluate_numpy(numpy, expression, x)
        error = prediction - target
        expected_cases.append(numpy.mean(error * error))
    expected = numpy.asarray(
        [expected_cases[idx % len(expected_cases)] for idx in range(actual.size)],
        dtype=numpy.float32,
    )
    maximum_error = float(numpy.max(numpy.abs(actual - expected)))
    if maximum_error > 2.0e-5:
        raise RuntimeError(
            f"fixed Kozax fitness validation failed: max_abs_error={maximum_error}")


def time_gpu(jax, function, warmups, min_repeats, min_seconds):
    result = function()
    jax.block_until_ready(result)
    for _ in range(warmups):
        result = function()
    jax.block_until_ready(result)

    times = []
    total = 0.0
    while (
        (len(times) < min_repeats or total < min_seconds)
        and len(times) < MAX_REPEATS
    ):
        begin = time.perf_counter()
        result = function()
        jax.block_until_ready(result)
        seconds = time.perf_counter() - begin
        times.append(seconds)
        total += seconds

    if not times:
        raise RuntimeError("Kozax timing produced no result")
    return min(times), statistics.median(times), len(times), result


def collect_case(
    numpy,
    jax,
    jax_numpy,
    strategy,
    population,
    x_host,
    target_host,
    shape,
    warmups,
    min_repeats,
    min_seconds,
    seed,
):
    x = jax_numpy.asarray(x_host)
    target = jax_numpy.asarray(target_host)
    if shape == "materialize":
        materialize = make_materialize_function(jax, strategy)
        function = lambda: materialize(population, x)
        execution_mode = "jit_tree_evaluator_vmap"
        notes = (
            f"profile={CORPUS['profile']};"
            "jit_warm;wall_clock_synchronized")
    else:
        function = lambda: strategy.jit_eval(population, (x, target))
        execution_mode = "jit_symbolic_regression_fitness"
        notes = (
            f"profile={CORPUS['profile']};jit_warm;"
            "native_mean_squared_error;wall_clock_synchronized")

    best, median, repeats, result = time_gpu(
        jax,
        function,
        warmups,
        min_repeats,
        min_seconds,
    )
    if shape == "materialize":
        validate_materialize(numpy, result, x_host)
    else:
        validate_fitness(numpy, result, x_host, target_host)

    flat_result = numpy.asarray(result).reshape(-1)
    checksum = float(flat_result[0] + flat_result[-1])
    asts = population.shape[0]
    rows = x_host.shape[0]
    return {
        "system": "kozax",
        "backend": "gpu_jax_runtime_dispatch",
        "shape": shape,
        "ast_mode": "alu",
        "corpus": CORPUS["name"],
        "corpus_hash": CORPUS["hash"],
        "seed": seed,
        "rows": rows,
        "asts": asts,
        "workers": 1,
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
        description="Collect Kozax JAX expression-backend runtime data.")
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
    parser.add_argument("--warmups", type=positive_int, default=2)
    parser.add_argument("--min-repeats", type=positive_int, default=5)
    parser.add_argument("--min-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=positive_int, default=1)
    parser.add_argument(
        "--max-materialize-bytes",
        type=positive_int,
        default=4 * 1024 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-estimated-working-bytes",
        type=positive_int,
        default=28 * 1024 * 1024 * 1024,
    )
    parser.add_argument(
        "--skip-oversized",
        action="store_true",
        help="report and skip cases rejected by the explicit memory limits",
    )
    args = parser.parse_args(argv)
    if args.min_seconds <= 0.0:
        parser.error("--min-seconds must be positive")
    for values, label in (
        (args.shapes, "shapes"),
        (args.rows, "rows"),
        (args.asts, "asts"),
    ):
        if len(values) != len(set(values)):
            parser.error(f"--{label} contains duplicates")
    if any(asts % 2 != 0 for asts in args.asts):
        parser.error("Kozax requires every --asts value to be even")
    return args


def main(argv=None):
    global CORPUS, INPUTS, EXPRESSIONS, MAX_NODES

    args = parse_args(argv)
    CORPUS = load_corpus(args.corpus)
    INPUTS = CORPUS["num_inputs"]
    EXPRESSIONS = CORPUS["expressions"]
    MAX_NODES = max(len(postorder(expression)) for expression in EXPRESSIONS)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from kozax.fitness_functions.SR_fitness_function import (
        SymbolicRegressionFitnessFunction,
    )
    from kozax.genetic_programming import GeneticProgramming

    if not any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Kozax baseline requires a JAX accelerator device")

    key_fields = ("shape", "rows", "asts")
    records, completed = resume_records(
        args.output,
        key_fields,
        {
            "system": ("kozax",),
            "backend": ("gpu_jax_runtime_dispatch",),
            "shape": args.shapes,
            "rows": args.rows,
            "asts": args.asts,
            "workers": (1,),
            "seed": (args.seed,),
            "corpus": (CORPUS["name"],),
            "corpus_hash": (CORPUS["hash"],),
        },
    )
    for asts in args.asts:
        strategy = make_strategy(
            jnp,
            GeneticProgramming,
            SymbolicRegressionFitnessFunction,
            asts,
        )
        population = make_population(np, jnp, strategy, asts)
        for rows in args.rows:
            x, target = make_data(np, rows, args.seed)
            for shape in args.shapes:
                key = tuple(str(value) for value in (shape, rows, asts))
                if key in completed:
                    print(
                        f"skip kozax {shape} rows={rows} asts={asts}",
                        flush=True,
                    )
                    continue
                estimated_working_bytes = estimate_working_bytes(rows, asts)
                if estimated_working_bytes > args.max_estimated_working_bytes:
                    message = (
                        "Kozax estimated JAX tree-evaluator working set would "
                        "exceed --max-estimated-working-bytes for "
                        f"rows={rows}, asts={asts}, "
                        f"estimated_bytes={estimated_working_bytes}")
                    if args.skip_oversized:
                        print(f"skip {message}", flush=True)
                        continue
                    raise RuntimeError(message)
                if (
                    shape == "materialize"
                    and rows * asts * 4 > args.max_materialize_bytes
                ):
                    message = (
                        "Kozax materialize output would exceed "
                        f"--max-materialize-bytes for rows={rows}, asts={asts}")
                    if args.skip_oversized:
                        print(f"skip {message}", flush=True)
                        continue
                    raise RuntimeError(message)
                record = collect_case(
                    np,
                    jax,
                    jnp,
                    strategy,
                    population,
                    x,
                    target,
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
                    f"mode={record['execution_mode']} row_evals_per_second="
                    f"{float(record['row_evals_per_second']):.3e}",
                    flush=True,
                )
    print(
        f"versions kozax={version('kozax')} jax={version('jax')} "
        f"jaxlib={version('jaxlib')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
