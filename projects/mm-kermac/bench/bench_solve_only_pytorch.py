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
import sys

import torch


def format_bytes(num_bytes: int) -> tuple[float, str]:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    idx = 0
    while value >= 1024.0 and idx + 1 < len(units):
        value /= 1024.0
        idx += 1
    return value, units[idx]


def format_time_ms(ms_value: float) -> tuple[float, str]:
    if ms_value >= 1.0:
        return ms_value, "ms"
    if ms_value >= 1e-3:
        return ms_value * 1e3, "us"
    return ms_value * 1e6, "ns"


def print_result(label: str, ms: float, n: int, rhs: int, batches: int) -> None:
    seconds = ms * 1e-3
    ms_per_batch = ms / float(batches)
    batches_per_s = float(batches) / seconds
    flops_per_batch = (float(n) * float(n) * float(n)) / 3.0 + 2.0 * float(n) * float(n) * float(rhs)
    gflops_s = (flops_per_batch * float(batches)) / seconds / 1e9
    total_time_value, total_time_unit = format_time_ms(ms)
    batch_time_value, batch_time_unit = format_time_ms(ms_per_batch)
    print(
        f"{label:<32s} {total_time_value:10.3f} {total_time_unit:>2s}  "
        f"{batch_time_value:10.3f} {batch_time_unit:>2s}/batch  "
        f"{batches_per_s:10.3f} batches/s  {gflops_s:10.3f} GFLOP/s"
    )


def benchmark_ms_excluding_setup(warmup: int, iters: int, setup, timed) -> float:
    warmup = max(warmup, 0)
    iters = max(iters, 1)

    for _ in range(warmup):
        setup()
        timed()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        setup()
        starts[i].record()
        timed()
        stops[i].record()

    stops[-1].synchronize()
    total_ms = sum(start.elapsed_time(stop) for start, stop in zip(starts, stops))
    return total_ms / float(iters)


def generate_spd_matrix(
    n: int,
    batches: int,
    regularizer: float,
    offdiag_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = torch.rand((batches, n, n), device=device, dtype=dtype, generator=generator)
    a = (a * 2.0 - 1.0) * offdiag_scale
    a = torch.triu(a, diagonal=1)
    a = a + a.transpose(-1, -2)
    diag = a.abs().sum(dim=-1) + (1.0 + regularizer)
    a.diagonal(dim1=-2, dim2=-1).copy_(diag)
    return a


def generate_rhs(
    n: int,
    rhs: int,
    batches: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand((batches, n, rhs), device=device, dtype=dtype, generator=generator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch CUDA solve-only benchmark")
    parser.add_argument("--N", type=int, default=2048, help="Matrix size (A is NxN)")
    parser.add_argument("--C", type=int, default=32, help="RHS columns / labels")
    parser.add_argument("--L", type=int, default=4, help="Batch size")
    parser.add_argument("--packed", choices=("lower", "upper"), default="lower", help="Triangle used by Cholesky")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument("--regularizer", type=float, default=1e-3, help="Extra diagonal regularizer")
    parser.add_argument("--offdiag-scale", type=float, default=5e-2, help="Off-diagonal random scale")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for matrix/RHS generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.N <= 0 or args.C <= 0 or args.L <= 0:
        print("Invalid dimensions: N,C,L must be > 0", file=sys.stderr)
        return 1
    if args.iters < 1:
        print("iters must be >= 1", file=sys.stderr)
        return 1
    if args.warmup < 0:
        print("warmup must be >= 0", file=sys.stderr)
        return 1
    if args.regularizer <= 0.0:
        print("regularizer must be > 0", file=sys.stderr)
        return 1
    if args.offdiag_scale < 0.0:
        print("offdiag-scale must be >= 0", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        return 77

    device = torch.device("cuda")
    dtype = torch.float32
    upper = args.packed == "upper"

    matrix_bytes = args.N * args.N * args.L * torch.finfo(dtype).bits // 8
    rhs_bytes = args.N * args.C * args.L * torch.finfo(dtype).bits // 8
    resident_bytes = 2 * matrix_bytes + 2 * rhs_bytes

    print("Solve-Only Benchmark (PyTorch)")
    print(
        f"N={args.N} C={args.C} L={args.L} packed={args.packed} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(
        f"regularizer={args.regularizer:.6g} offdiag_scale={args.offdiag_scale:.6g} "
        f"seed={args.seed} dtype=float32"
    )
    resident_value, resident_unit = format_bytes(resident_bytes)
    print(
        f"Resident tensor bytes={resident_bytes} "
        f"({resident_value:.3f} {resident_unit})"
    )
    print()

    a_orig = generate_spd_matrix(
        args.N,
        args.L,
        args.regularizer,
        args.offdiag_scale,
        args.seed,
        device,
        dtype,
    )
    b_orig = generate_rhs(args.N, args.C, args.L, args.seed + 1, device, dtype)
    a = torch.empty_like(a_orig)
    b = torch.empty_like(b_orig)
    torch.cuda.synchronize()

    state: dict[str, torch.Tensor] = {}

    def setup() -> None:
        a.copy_(a_orig)
        b.copy_(b_orig)

    def timed() -> None:
        factor, info = torch.linalg.cholesky_ex(a, upper=upper, check_errors=False)
        solution = torch.cholesky_solve(b, factor, upper=upper)
        state["factor"] = factor
        state["info"] = info
        state["solution"] = solution

    ms = benchmark_ms_excluding_setup(args.warmup, args.iters, setup, timed)
    print_result("pytorch_cholesky_solve", ms, args.N, args.C, args.L)

    setup()
    timed()
    torch.cuda.synchronize()
    info = state["info"]
    if torch.any(info != 0):
        bad = torch.nonzero(info != 0, as_tuple=False)[0].tolist()
        batch_idx = int(bad[0])
        info_value = int(info[batch_idx].item())
        print(
            f"pytorch_cholesky_solve failed: info[{batch_idx}]={info_value}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
