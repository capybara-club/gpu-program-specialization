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

"""Map a live EvoGP population to Secant bytecode and compare execution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np


SUPPORTED_FUNCTION_NAMES = (
    "+",
    "-",
    "*",
    "max",
    "min",
    "neg",
    "abs",
    "sin",
    "cos",
    "tanh",
)


@dataclass(frozen=True)
class MappedTree:
    expression: str
    bytecode: bytes
    node_count: int


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _interop_executable_find(repo_root: Path) -> Path:
    candidates = (
        repo_root / "build" / "secant_evogp_interop",
        repo_root.parent / "build" / "secant_evogp_interop",
        repo_root / "build-bench" / "secant_evogp_interop",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "secant_evogp_interop was not found; configure with "
        "-DSECANT_BUILD_BENCHMARKS=ON and build that target"
    )


def _tree_map(secant_ast, func, ntype, values, types, subtree_sizes) -> MappedTree:
    operation_map = {
        func.ADD: (2, lambda a: a[0] + a[1], lambda a: f"({a[0]} + {a[1]})"),
        func.SUB: (2, lambda a: a[0] - a[1], lambda a: f"({a[0]} - {a[1]})"),
        func.MUL: (2, lambda a: a[0] * a[1], lambda a: f"({a[0]} * {a[1]})"),
        func.MAX: (2, lambda a: secant_ast.maximum(a[0], a[1]), lambda a: f"max({a[0]}, {a[1]})"),
        func.MIN: (2, lambda a: secant_ast.minimum(a[0], a[1]), lambda a: f"min({a[0]}, {a[1]})"),
        func.NEG: (1, lambda a: -a[0], lambda a: f"(-{a[0]})"),
        func.ABS: (1, lambda a: secant_ast.absolute(a[0]), lambda a: f"abs({a[0]})"),
        func.SIN: (1, lambda a: secant_ast.sin(a[0]), lambda a: f"sin({a[0]})"),
        func.COS: (1, lambda a: secant_ast.cos(a[0]), lambda a: f"cos({a[0]})"),
        func.TANH: (1, lambda a: secant_ast.tanh(a[0]), lambda a: f"tanh({a[0]})"),
    }
    root_size = int(subtree_sizes[0])

    if root_size <= 0 or root_size > len(values):
        raise ValueError(f"invalid EvoGP root subtree size {root_size}")

    def decode(node_index: int):
        if node_index >= root_size:
            raise ValueError("EvoGP prefix tree ended inside an operation")
        node_type = int(types[node_index]) & ntype.TYPE_MASK
        node_size = int(subtree_sizes[node_index])
        if node_size <= 0 or node_index + node_size > root_size:
            raise ValueError(f"invalid subtree size {node_size} at node {node_index}")

        if node_type == ntype.VAR:
            input_index = int(values[node_index])
            if input_index < 0 or input_index >= 128:
                raise ValueError(f"EvoGP variable index {input_index} is outside Secant's range")
            if node_size != 1:
                raise ValueError(f"variable node {node_index} has subtree size {node_size}")
            return secant_ast.input(input_index), f"x{input_index}", node_index + 1

        if node_type == ntype.CONST:
            value = np.float32(values[node_index])
            bits = int(value.view(np.uint32))
            if node_size != 1:
                raise ValueError(f"constant node {node_index} has subtree size {node_size}")
            return secant_ast.constant_bits(bits), repr(float(value)), node_index + 1

        expected_arity = {
            ntype.UFUNC: 1,
            ntype.BFUNC: 2,
            ntype.TFUNC: 3,
        }.get(node_type)
        if expected_arity is None:
            raise ValueError(f"unsupported EvoGP node type {node_type} at node {node_index}")

        function = int(values[node_index])
        operation = operation_map.get(function)
        if operation is None:
            raise ValueError(f"unsupported EvoGP function opcode {function} at node {node_index}")
        arity, expression_build, text_build = operation
        if arity != expected_arity:
            raise ValueError(
                f"EvoGP function opcode {function} has node arity {expected_arity}, expected {arity}"
            )

        cursor = node_index + 1
        arguments = []
        argument_text = []
        for _ in range(arity):
            argument, text, cursor = decode(cursor)
            arguments.append(argument)
            argument_text.append(text)
        if cursor - node_index != node_size:
            raise ValueError(
                f"subtree size mismatch at node {node_index}: metadata={node_size} decoded={cursor - node_index}"
            )
        return expression_build(arguments), text_build(argument_text), cursor

    expression, expression_text, end = decode(0)
    if end != root_size:
        raise ValueError(f"EvoGP root consumed {end} nodes but metadata reports {root_size}")
    program = secant_ast.Program(expression)
    secant_ast.validate_program(program)
    return MappedTree(expression_text, program.bytecode, root_size)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evogp-repo",
        type=Path,
        default=Path("/home/cdurham/code/evogp_trial/evogp"),
    )
    parser.add_argument("--interop-executable", type=Path)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "scratch" / "evogp_secant_interop")
    parser.add_argument("--population", type=_positive, default=128)
    parser.add_argument("--rows", type=_positive, default=1024)
    parser.add_argument("--inputs", type=_positive, default=6)
    parser.add_argument("--generations", type=_positive, default=4)
    parser.add_argument("--max-tree-len", type=_positive, default=32)
    parser.add_argument("--max-layers", type=_positive, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[2]
    evogp_source = args.evogp_repo.resolve() / "src"
    if not evogp_source.is_dir():
        raise RuntimeError(f"EvoGP source directory does not exist: {evogp_source}")
    sys.path.insert(0, str(evogp_source))

    import torch
    from evogp.algorithm import DefaultCrossover, DefaultMutation, DefaultSelection, GeneticProgramming
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor
    from evogp.tree.utils import Func, NType

    secant_ast = _load_module("secant_ast_interop", repo_root / "python" / "secant" / "ast.py")
    executable = args.interop_executable or _interop_executable_find(repo_root)
    if not executable.is_file():
        raise RuntimeError(f"interop executable does not exist: {executable}")
    if args.inputs > 128:
        raise ValueError("Secant supports at most 128 static input columns")
    if args.population + args.inputs > 256:
        raise ValueError("this one-kernel interoperability template requires population + inputs <= 256")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    input_rows = torch.rand(args.rows, args.inputs, device="cuda", dtype=torch.float32) * 4.0 - 2.0
    target = input_rows[:, 0].clone()
    if args.inputs > 2:
        target = target + input_rows[:, 1] * input_rows[:, 2]
    if args.inputs > 3:
        target = target - 0.75 * input_rows[:, 3]
    if args.inputs > 4:
        target = target + 0.1 * input_rows[:, 4]

    descriptor = GenerateDescriptor(
        max_tree_len=args.max_tree_len,
        input_len=args.inputs,
        output_len=1,
        using_funcs=list(SUPPORTED_FUNCTION_NAMES),
        max_layer_cnt=args.max_layers,
        const_samples=[-1.5, -1.0, -0.75, -0.25, 0.25, 0.75, 1.0, 1.5],
        const_prob=0.3,
        out_prob=0.0,
        layer_leaf_prob=0.3,
    )
    algorithm = GeneticProgramming(
        initial_forest=Forest.random_generate(args.population, descriptor),
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(0.2, descriptor.update(max_layer_cnt=max(2, args.max_layers - 1))),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
        enable_pareto_front=False,
    )
    problem = SymbolicRegression(
        datapoints=input_rows,
        labels=target.reshape(-1, 1),
        execute_mode="hybrid parallel",
    )

    fitness_history = []
    fitness = None
    for generation in range(args.generations):
        fitness = problem.evaluate(algorithm.forest)
        torch.cuda.synchronize()
        fitness_history.append(float(torch.max(fitness).item()))
        if generation + 1 < args.generations:
            algorithm.step(fitness)
    assert fitness is not None

    forest = algorithm.forest
    evogp_output = forest.batch_forward(input_rows)[:, :, 0].contiguous()
    torch.cuda.synchronize()
    values = forest.batch_node_value.detach().cpu().numpy()
    types = forest.batch_node_type.detach().cpu().numpy()
    subtree_sizes = forest.batch_subtree_size.detach().cpu().numpy()
    fitness_cpu = fitness.detach().cpu().numpy()

    mapped = [
        _tree_map(secant_ast, Func, NType, values[index], types[index], subtree_sizes[index])
        for index in range(args.population)
    ]
    packed_programs, offsets = secant_ast.pack_programs(tree.bytecode for tree in mapped)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    programs_path = args.output_dir / "programs.bin"
    offsets_path = args.output_dir / "program_offsets.u64"
    input_path = args.output_dir / "input_columns.f32"
    expected_path = args.output_dir / "evogp_output.f32"
    mapping_path = args.output_dir / "population_map.json"

    programs_path.write_bytes(packed_programs)
    np.asarray(offsets, dtype="<u8").tofile(offsets_path)
    input_rows.T.contiguous().cpu().numpy().astype("<f4", copy=False).tofile(input_path)
    evogp_output.cpu().numpy().astype("<f4", copy=False).tofile(expected_path)

    order = np.argsort(-fitness_cpu)
    mapping = {
        "format": "secant_evoGP_population_map_v1",
        "seed": args.seed,
        "rows": args.rows,
        "inputs": args.inputs,
        "population": args.population,
        "generations": args.generations,
        "max_tree_len": args.max_tree_len,
        "operators": list(SUPPORTED_FUNCTION_NAMES),
        "fitness_history": fitness_history,
        "population_entries": [
            {
                "ast_index": index,
                "fitness": float(fitness_cpu[index]),
                "evoGP_expression": mapped[index].expression,
                "evoGP_node_count": mapped[index].node_count,
                "evoGP_node_values": [
                    float(value) for value in values[index, : mapped[index].node_count]
                ],
                "evoGP_node_types": [
                    int(value) for value in types[index, : mapped[index].node_count]
                ],
                "evoGP_subtree_sizes": [
                    int(value) for value in subtree_sizes[index, : mapped[index].node_count]
                ],
                "secant_bytecode_hex": mapped[index].bytecode.hex(),
            }
            for index in range(args.population)
        ],
    }
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    print(
        f"evogp population={args.population} rows={args.rows} inputs={args.inputs} "
        f"generations={args.generations} best_fitness={fitness_history[-1]:.9g}"
    )
    for rank, ast_index in enumerate(order[: min(5, len(order))]):
        tree = mapped[int(ast_index)]
        print(
            f"rank[{rank}] ast={int(ast_index)} fitness={float(fitness_cpu[ast_index]):.9g} "
            f"nodes={tree.node_count} expression={tree.expression}"
        )
    print(f"population_map={mapping_path}")
    sys.stdout.flush()

    command = [
        str(executable),
        str(programs_path),
        str(offsets_path),
        str(input_path),
        str(expected_path),
        str(args.population),
        str(args.inputs),
        str(args.rows),
    ]
    completed = subprocess.run(
        command,
        check=False,
        env={**dict(os.environ), "CUDA_MODULE_LOADING": "EAGER"},
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
