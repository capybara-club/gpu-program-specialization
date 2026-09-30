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
import sys

from backend_csv import BACKENDS
import compile_matrix_graph
import runtime_packing_graph
import runtime_sse_surface_graph
import runtime_tile_graph


def output_path(output_dir, prefix, name, output_format):
    return output_dir / f"{prefix}_{name}.{output_format}"


def draw_graphs(
    compile_csvs,
    runtime_packing_csvs,
    runtime_tile_csvs,
    runtime_surface_csvs,
    output_dir,
    prefix,
    output_format,
    backends,
    log_scale,
):
    outputs = []

    if compile_csvs:
        path = output_path(output_dir, prefix, "compile", output_format)
        compile_matrix_graph.draw_graph(
            compile_csvs,
            path,
            requested_backends=backends,
            log_scale=log_scale,
        )
        outputs.append(path)

    if runtime_packing_csvs:
        for shape in runtime_packing_graph.KERNEL_SHAPES:
            path = output_path(
                output_dir,
                prefix,
                f"runtime_{shape}_packing",
                output_format,
            )
            runtime_packing_graph.draw_graph(
                runtime_packing_csvs,
                path,
                shape,
                requested_backends=backends,
                log_scale=log_scale,
            )
            outputs.append(path)

    if runtime_tile_csvs:
        path = output_path(output_dir, prefix, "runtime_tile", output_format)
        runtime_tile_graph.draw_graph(
            runtime_tile_csvs,
            path,
            requested_backends=backends,
            log_scale=log_scale,
        )
        outputs.append(path)

    if runtime_surface_csvs:
        path = output_path(output_dir, prefix, "runtime_sse_surface", output_format)
        runtime_sse_surface_graph.draw_graph(
            runtime_surface_csvs,
            path,
            requested_backends=backends,
            log_scale=log_scale,
        )
        outputs.append(path)

    if not outputs:
        raise RuntimeError("no input CSVs were supplied")

    return outputs


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate SECANT compile and runtime graphs.")
    parser.add_argument("--compile-csv", type=Path, nargs="+")
    parser.add_argument("--runtime-packing-csv", type=Path, nargs="+")
    parser.add_argument("--runtime-tile-csv", type=Path, nargs="+")
    parser.add_argument("--runtime-surface-csv", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="secant")
    parser.add_argument(
        "--format",
        choices=("svg", "png", "pdf"),
        default="svg",
    )
    parser.add_argument(
        "--backends",
        choices=BACKENDS,
        nargs="+",
    )
    parser.add_argument("--log-scale", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outputs = draw_graphs(
        args.compile_csv,
        args.runtime_packing_csv,
        args.runtime_tile_csv,
        args.runtime_surface_csv,
        args.output_dir,
        args.prefix,
        args.format,
        args.backends,
        args.log_scale,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
