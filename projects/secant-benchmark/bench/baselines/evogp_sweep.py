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
from pathlib import Path
import statistics
import sys

from common import positive_int, resume_records, write_csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from corpus import (
    DEFAULT_CORPUS,
    load_corpus,
    lower_safe_operations,
    make_torch_input,
)


SHAPES = ("materialize", "sse")
EXECUTE_MODES = ("hybrid parallel", "data parallel", "tree parallel", "auto")
MAX_REPEATS = 256
CORPUS = None
INPUTS = 0
EXPRESSIONS = ()
TREE_LEN = 0


def add_evogp_to_path(repo):
    source = repo / "src"
    if not source.is_dir():
        raise RuntimeError(f"EvoGP source directory does not exist: {source}")
    sys.path.insert(0, str(source))


def encode_expression(expression, values, types, subtree_sizes, func, ntype):
    start = len(values)
    if isinstance(expression, int):
        values.append(float(expression))
        types.append(ntype.VAR)
        subtree_sizes.append(1)
        return

    if expression[0] == "constant":
        values.append(float(expression[1]))
        types.append(ntype.CONST)
        subtree_sizes.append(1)
        return

    operation, *arguments = expression
    values.append(float({
        "add": func.ADD,
        "sub": func.SUB,
        "mul": func.MUL,
        "div": func.DIV,
        "min": func.MIN,
        "max": func.MAX,
        "neg": func.NEG,
        "abs": func.ABS,
        "sin": func.SIN,
        "cos": func.COS,
        "sqrt": func.SQRT,
        "tanh": func.TANH,
    }[operation]))
    types.append(ntype.UFUNC if len(arguments) == 1 else ntype.BFUNC)
    subtree_sizes.append(0)
    for argument in arguments:
        encode_expression(argument, values, types, subtree_sizes, func, ntype)
    subtree_sizes[start] = len(values) - start


def make_forest(torch, forest_type, func, ntype, asts):
    global TREE_LEN

    encoded = []
    for expression in EXPRESSIONS[:min(asts, len(EXPRESSIONS))]:
        values = []
        types = []
        subtree_sizes = []
        encode_expression(
            lower_safe_operations(expression),
            values,
            types,
            subtree_sizes,
            func,
            ntype,
        )
        if subtree_sizes[0] != len(values):
            raise RuntimeError("the encoded EvoGP expression has an invalid root")
        encoded.append((values, types, subtree_sizes))

    TREE_LEN = max(len(case[0]) for case in encoded) + 1
    if TREE_LEN > 1024:
        raise RuntimeError("the lowered EvoGP expression exceeds MAX_STACK")
    for values, types, subtree_sizes in encoded:
        padding = TREE_LEN - len(values)
        values.extend([0.0] * padding)
        types.extend([ntype.VAR] * padding)
        subtree_sizes.extend([0] * padding)

    values = torch.tensor(
        [encoded[index % len(encoded)][0] for index in range(asts)],
        dtype=torch.float32,
        device="cuda",
    )
    types = torch.tensor(
        [encoded[index % len(encoded)][1] for index in range(asts)],
        dtype=torch.int16,
        device="cuda",
    )
    subtree_sizes = torch.tensor(
        [encoded[index % len(encoded)][2] for index in range(asts)],
        dtype=torch.int16,
        device="cuda",
    )
    return forest_type(INPUTS, 1, values, types, subtree_sizes)


def evaluate_expression(torch, expression, x):
    if isinstance(expression, int):
        return x[:, expression]
    operation = expression[0]
    if operation == "constant":
        return torch.full_like(x[:, 0], float(expression[1]))
    arguments = [
        evaluate_expression(torch, argument, x)
        for argument in expression[1:]
    ]
    if operation == "add":
        return arguments[0] + arguments[1]
    if operation == "sub":
        return arguments[0] - arguments[1]
    if operation == "mul":
        return arguments[0] * arguments[1]
    if operation == "div":
        return arguments[0] / arguments[1]
    if operation == "min":
        return torch.minimum(arguments[0], arguments[1])
    if operation == "max":
        return torch.maximum(arguments[0], arguments[1])
    if operation == "safe_div":
        return (
            arguments[0] * arguments[1] /
            (arguments[1] * arguments[1] + 0.01)
        )
    if operation == "neg":
        return -arguments[0]
    if operation == "abs":
        return torch.abs(arguments[0])
    if operation == "sin":
        return torch.sin(arguments[0])
    if operation == "cos":
        return torch.cos(arguments[0])
    if operation == "sqrt":
        return torch.sqrt(arguments[0])
    if operation == "tanh":
        return torch.tanh(arguments[0])
    if operation == "safe_sqrt":
        return torch.sqrt(torch.abs(arguments[0]) + 0.01)
    if operation == "safe_rsqrt":
        return torch.rsqrt(torch.abs(arguments[0]) + 0.01)
    raise RuntimeError(f"unsupported fixed expression operation: {operation}")


def make_data(torch, rows, seed):
    x = make_torch_input(torch, CORPUS, rows, seed, "cuda")
    return x, evaluate_expression(
        torch,
        EXPRESSIONS[CORPUS["target_expression"]],
        x,
    ).reshape(-1, 1)


def validate_forest(torch, forest, x):
    check_rows = min(257, x.shape[0])
    outputs = forest.batch_forward(x[:check_rows])
    relative_tolerance = (
        2.0e-3 if CORPUS["profile"] == "safe-math" else 2.0e-4)
    for ast_idx in range(forest.pop_size):
        expected = evaluate_expression(
            torch,
            EXPRESSIONS[ast_idx % len(EXPRESSIONS)],
            x[:check_rows],
        )
        maximum_error = torch.max(
            torch.abs(outputs[ast_idx, :, 0] - expected) /
            (1.0 + torch.abs(expected))).item()
        if maximum_error > relative_tolerance:
            raise RuntimeError(
                "fixed EvoGP forest validation failed: "
                f"ast={ast_idx} max_relative_error={maximum_error} "
                f"relative_tolerance={relative_tolerance}")


def time_gpu(torch, function, warmups, min_repeats, min_seconds):
    for _ in range(warmups):
        result = function()
    torch.cuda.synchronize()

    times = []
    total = 0.0
    result = None
    while (
        (len(times) < min_repeats or total < min_seconds)
        and len(times) < MAX_REPEATS
    ):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) * 1.0e-3
        times.append(seconds)
        total += seconds

    if result is None:
        raise RuntimeError("EvoGP timing produced no result")
    flat_result = result.reshape(-1)
    checksum = float(flat_result[0].item() + flat_result[-1].item())
    return min(times), statistics.median(times), len(times), checksum


def estimate_materialize_bytes(rows, asts):
    node_bytes = TREE_LEN * (4 + 2 + 2)
    return rows * asts * (INPUTS * 4 + node_bytes + 4)


def collect_case(
    torch,
    symbolic_regression,
    forest,
    x,
    target,
    shape,
    execute_mode,
    warmups,
    min_repeats,
    min_seconds,
    seed,
):
    if shape == "materialize":
        function = lambda: forest.batch_forward(x)
        execution_mode = "batch_forward_public"
        notes = (
            f"profile={CORPUS['profile']};"
            "includes_evoGP_public_batch_expansion;"
            f"validation_rtol="
            f"{2.0e-3 if CORPUS['profile'] == 'safe-math' else 2.0e-4:g}")
    else:
        problem = symbolic_regression(
            datapoints=x,
            labels=target,
            execute_mode=execute_mode,
        )
        function = lambda: problem.evaluate(forest)
        execution_mode = execute_mode.replace(" ", "_")
        notes = (
            f"profile={CORPUS['profile']};native_sr_fitness_mse;"
            f"validation_rtol="
            f"{2.0e-3 if CORPUS['profile'] == 'safe-math' else 2.0e-4:g}")

    best, median, repeats, checksum = time_gpu(
        torch,
        function,
        warmups,
        min_repeats,
        min_seconds,
    )
    asts = forest.pop_size
    rows = x.shape[0]
    return {
        "system": "evogp",
        "backend": "gpu_runtime_dispatch",
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
        description="Collect reproducible EvoGP runtime-dispatch data.")
    parser.add_argument(
        "--evogp-repo",
        type=Path,
        default=Path("/home/cdurham/code/evogp_trial/evogp"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--shapes", choices=SHAPES, nargs="+", default=list(SHAPES))
    parser.add_argument(
        "--execute-modes",
        choices=EXECUTE_MODES,
        nargs="+",
        default=["hybrid parallel"],
    )
    parser.add_argument(
        "--rows",
        type=positive_int,
        nargs="+",
        default=[1_024, 16_384],
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
        "--skip-oversized",
        action="store_true",
        help="report and skip materialize cases rejected by the memory limit",
    )
    args = parser.parse_args(argv)
    if args.min_seconds <= 0.0:
        parser.error("--min-seconds must be positive")
    for values, label in (
        (args.shapes, "shapes"),
        (args.execute_modes, "execute-modes"),
        (args.rows, "rows"),
        (args.asts, "asts"),
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
    add_evogp_to_path(args.evogp_repo)

    import torch
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest
    from evogp.tree.utils import Func, NType

    if not torch.cuda.is_available():
        raise RuntimeError(
            "EvoGP requires a GPU accelerator through PyTorch's CUDA API")

    accelerator = "rocm" if torch.version.hip is not None else "cuda"
    print(
        f"evogp accelerator={accelerator} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    key_fields = ("shape", "rows", "asts", "execution_mode")
    records, completed = resume_records(
        args.output,
        key_fields,
        {
            "system": ("evogp",),
            "backend": ("gpu_runtime_dispatch",),
            "shape": args.shapes,
            "rows": args.rows,
            "asts": args.asts,
            "workers": (1,),
            "seed": (args.seed,),
            "corpus": (CORPUS["name"],),
            "corpus_hash": (CORPUS["hash"],),
        },
    )
    for rows in args.rows:
        x, target = make_data(torch, rows, args.seed)
        for asts in args.asts:
            forest = make_forest(torch, Forest, Func, NType, asts)
            validate_forest(torch, forest, x)
            for shape in args.shapes:
                if (
                    shape == "materialize"
                    and estimate_materialize_bytes(rows, asts)
                    > args.max_materialize_bytes
                ):
                    message = (
                        "EvoGP public materialize path would exceed "
                        f"--max-materialize-bytes for rows={rows}, asts={asts}")
                    if args.skip_oversized:
                        print(f"skip {message}", flush=True)
                        continue
                    raise RuntimeError(message)
                modes = args.execute_modes if shape == "sse" else (None,)
                for execute_mode in modes:
                    execution_mode = (
                        execute_mode.replace(" ", "_")
                        if execute_mode is not None
                        else "batch_forward_public"
                    )
                    key = tuple(str(value) for value in (
                        shape,
                        rows,
                        asts,
                        execution_mode,
                    ))
                    if key in completed:
                        print(
                            f"skip evogp {shape} rows={rows} asts={asts} "
                            f"mode={execution_mode}",
                            flush=True,
                        )
                        continue
                    record = collect_case(
                        torch,
                        SymbolicRegression,
                        forest,
                        x,
                        target,
                        shape,
                        execute_mode,
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
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
