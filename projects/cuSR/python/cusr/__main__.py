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

import argparse
from pathlib import Path

from .compiler import compile_tile_static_mse_cubin, make_tile_static_mse_instantiations


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile cuSR tile-static MSE template specializations to a cubin")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--kernels", type=int, default=8)
    parser.add_argument("--ast-capacity", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--tile-rows", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--cta-threads", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--arch", help="target architecture such as sm_120; defaults to the current device")
    parser.add_argument("--print-log", action="store_true")
    args = parser.parse_args()

    instantiations = make_tile_static_mse_instantiations(
        args.kernels,
        ast_capacity=args.ast_capacity,
        tile_rows=args.tile_rows,
        threads_per_cta=args.cta_threads,
    )
    result = compile_tile_static_mse_cubin(instantiations, arch=args.arch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result.cubin)
    print(f"cubin={args.output} bytes={result.size} kernels={len(instantiations)}")
    for expression, lowered_name in zip(result.name_expressions, result.lowered_names, strict=True):
        print(f"{expression} -> {lowered_name}")
    if args.print_log and result.log:
        print(result.log, end="" if result.log.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
