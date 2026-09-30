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
from dataclasses import dataclass
import statistics
import time

import numpy as np
import torch

import cusr


_ALU_OPS = (cusr.ADD, cusr.MUL, cusr.MIN, cusr.MAX)
_MUFU_OPS = (cusr.SIN, cusr.COS, cusr.SQRT, cusr.RCP, cusr.EX2, cusr.LG2, cusr.RSQRT, cusr.TANH)


@dataclass(frozen=True, slots=True)
class RunTiming:
    stream_count: int
    host_submit_ms: float
    gpu_elapsed_ms: float
    synchronized_ms: float


def _positive_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _stream_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("stream counts must be comma-separated integers") from error
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("stream counts must be positive")
    return tuple(dict.fromkeys(counts))


def _random_program(rng: np.random.Generator, mode: str) -> tuple[np.uint64, ...]:
    nodes: list[list[np.uint64]] = []
    input_order = rng.permutation(8)
    for input_idx in input_order:
        node = [cusr.encode_input(int(input_idx))]
        if mode == "mufu":
            node.append(_MUFU_OPS[int(rng.integers(len(_MUFU_OPS)))])
        nodes.append(node)

    while len(nodes) > 1:
        next_nodes: list[list[np.uint64]] = []
        for node_idx in range(0, len(nodes), 2):
            next_nodes.append(
                nodes[node_idx]
                + nodes[node_idx + 1]
                + [_ALU_OPS[int(rng.integers(len(_ALU_OPS)))]]
            )
        nodes = next_nodes

    return tuple((*nodes[0], cusr.RETURN))


def _make_programs(count: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    return cusr.pack_programs(tuple(_random_program(rng, mode) for _ in range(count)))


def _make_settings(
    num_settings: int,
    num_columns: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    masks = rng.integers(0, 256, size=num_settings, dtype=np.uint8)
    constants = rng.uniform(0.5, 1.5, size=(num_settings, 8)).astype(np.float32)
    words = constants.view(np.uint32).copy()

    for input_idx in range(8):
        active = ((masks >> input_idx) & 1).astype(bool)
        words[active, input_idx] = rng.integers(
            0,
            num_columns,
            size=int(np.count_nonzero(active)),
            dtype=np.uint32,
        )
    return masks, words


def _time_patching(
    layout: object,
    programs: np.ndarray,
    cubin: bytearray,
    epilogue: cusr.PatchEpilogue,
    iterations: int,
) -> tuple[object, tuple[float, ...]]:
    stats = cusr.patch_cubin_in_place(layout, programs, cubin, epilogue=epilogue)
    samples = []
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        stats = cusr.patch_cubin_in_place(layout, programs, cubin, epilogue=epilogue)
        samples.append((time.perf_counter_ns() - start_ns) * 1.0e-6)
    return stats, tuple(samples)


def _record_worker_completion(
    producer: torch.cuda.Stream,
    workers: tuple[torch.cuda.Stream, ...],
    done_events: tuple[torch.cuda.Event, ...],
) -> None:
    for worker, done in zip(workers, done_events, strict=True):
        with torch.cuda.stream(worker):
            done.record()
    for done in done_events:
        producer.wait_event(done)


def _time_runtime(
    module: object,
    family: str,
    x: torch.Tensor,
    target: torch.Tensor | None,
    masks: torch.Tensor,
    words: torch.Tensor,
    asts_per_kernel: int,
    output: torch.Tensor,
    workers: tuple[torch.cuda.Stream, ...],
    warmup_iterations: int,
    run_iterations: int,
) -> RunTiming:
    producer = torch.cuda.current_stream(x.device)
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    done_events = tuple(torch.cuda.Event() for _ in workers)

    def submit() -> None:
        if family == "mse":
            assert target is not None
            module(
                x,
                target,
                masks,
                words,
                asts_per_kernel=asts_per_kernel,
                output_sse=output,
                streams=workers,
            )
        else:
            module(
                x,
                masks,
                words,
                asts_per_kernel=asts_per_kernel,
                output_values=output,
                streams=workers,
            )

    for _ in range(warmup_iterations):
        submit()
        _record_worker_completion(producer, workers, done_events)
        torch.cuda.synchronize(x.device)

    host_samples = []
    gpu_samples = []
    synchronized_samples = []
    for _ in range(run_iterations):
        torch.cuda.synchronize(x.device)
        synchronized_start_ns = time.perf_counter_ns()
        start_event.record(producer)
        host_start_ns = time.perf_counter_ns()
        submit()
        host_samples.append((time.perf_counter_ns() - host_start_ns) * 1.0e-6)
        _record_worker_completion(producer, workers, done_events)
        stop_event.record(producer)
        stop_event.synchronize()
        synchronized_samples.append((time.perf_counter_ns() - synchronized_start_ns) * 1.0e-6)
        gpu_samples.append(start_event.elapsed_time(stop_event))

    return RunTiming(
        stream_count=len(workers),
        host_submit_ms=statistics.median(host_samples),
        gpu_elapsed_ms=statistics.median(gpu_samples),
        synchronized_ms=statistics.median(synchronized_samples),
    )


def _validate_output(
    family: str,
    programs: np.ndarray,
    x: np.ndarray,
    target: np.ndarray | None,
    masks: np.ndarray,
    words: np.ndarray,
    output: torch.Tensor,
    check_asts: int,
    check_settings: int,
    check_rows: int,
) -> tuple[float, float]:
    selected_programs = programs[:check_asts]
    selected_masks = masks[:check_settings]
    selected_words = words[:check_settings]

    if family == "mse":
        assert target is not None
        expected = cusr.evaluate_sse(selected_programs, x, target, selected_masks, selected_words)
        actual = output[:check_asts, :check_settings].cpu().numpy()
    else:
        expected = cusr.evaluate_values(
            selected_programs,
            x,
            selected_masks,
            selected_words,
            num_rows=check_rows,
        )
        actual = output[:check_asts, :check_settings, :check_rows].cpu().numpy()

    if family == "mse":
        np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=3.0e-2)
    else:
        np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=3.0e-5)

    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), np.float32(1.0e-6))
    return float(np.max(absolute)), float(np.max(relative))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Python-driven cuSR cubin patching, module loading, and kernel submission."
    )
    parser.add_argument("--family", choices=("mse", "eval"), default="mse")
    parser.add_argument("--kernels", type=_positive_int, default=128)
    parser.add_argument("--ast-capacity", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--asts-per-kernel", type=_positive_int, default=32)
    parser.add_argument("--rows", type=_positive_int)
    parser.add_argument("--settings", type=_positive_int)
    parser.add_argument("--columns", type=_positive_int, default=32)
    parser.add_argument("--tile-rows", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--threads", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--streams", type=_stream_counts, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--ast-mode", choices=("alu", "mufu"), default="alu")
    parser.add_argument("--patch-iterations", type=_positive_int, default=10)
    parser.add_argument("--warmup-iterations", type=_positive_int, default=2)
    parser.add_argument("--run-iterations", type=_positive_int, default=5)
    parser.add_argument("--check-asts", type=_positive_int, default=2)
    parser.add_argument("--check-settings", type=_positive_int, default=4)
    parser.add_argument("--check-rows", type=_positive_int, default=256)
    parser.add_argument("--max-output-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0xC057BEEF)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.asts_per_kernel > args.ast_capacity:
        raise SystemExit("--asts-per-kernel must not exceed --ast-capacity")
    if not 1 <= args.columns <= 32:
        raise SystemExit("--columns must be in [1, 32]")
    if args.max_output_gib <= 0.0:
        raise SystemExit("--max-output-gib must be positive")

    rows = args.rows if args.rows is not None else (4096 if args.family == "mse" else 1024)
    settings = args.settings if args.settings is not None else (4096 if args.family == "mse" else 16)
    total_asts = args.kernels * args.asts_per_kernel
    output_elements = total_asts * settings * (1 if args.family == "mse" else rows)
    output_gib = output_elements * np.dtype(np.float32).itemsize / (1024.0 ** 3)
    if output_gib > args.max_output_gib:
        raise SystemExit(
            f"requested output is {output_gib:.3f} GiB; increase --max-output-gib or reduce the shape"
        )

    torch.cuda.set_device(args.device)
    capability_major, capability_minor = torch.cuda.get_device_capability(args.device)
    if capability_major not in (8, 9, 10, 12):
        raise SystemExit(f"unsupported CUDA capability {capability_major}.{capability_minor}")
    arch = f"sm_{capability_major}{capability_minor}"
    rng = np.random.default_rng(args.seed)

    programs = _make_programs(total_asts, args.ast_mode, rng)
    masks, words = _make_settings(settings, args.columns, rng)
    x = rng.uniform(0.5, 1.5, size=(args.columns, rows)).astype(np.float32)
    target = rng.uniform(-1.0, 1.0, size=rows).astype(np.float32) if args.family == "mse" else None

    x_cuda = torch.from_numpy(x).to(device=f"cuda:{args.device}")
    masks_cuda = torch.from_numpy(masks).to(device=f"cuda:{args.device}")
    words_cuda = torch.from_numpy(words).to(device=f"cuda:{args.device}")
    target_cuda = None if target is None else torch.from_numpy(target).to(device=f"cuda:{args.device}")
    if args.family == "mse":
        output = torch.empty((total_asts, settings), dtype=torch.float32, device=x_cuda.device)
    else:
        output = torch.empty((total_asts, settings, rows), dtype=torch.float32, device=x_cuda.device)
    torch.cuda.synchronize(args.device)

    if args.family == "mse":
        instantiations = cusr.make_tile_static_mse_instantiations(
            args.kernels,
            ast_capacity=args.ast_capacity,
            tile_rows=args.tile_rows,
            threads_per_cta=args.threads,
        )
        compile_start_ns = time.perf_counter_ns()
        compiled = cusr.compile_tile_static_mse_cubin(instantiations, arch=arch)
        compile_ms = (time.perf_counter_ns() - compile_start_ns) * 1.0e-6
        epilogue = cusr.PatchEpilogue.SSE
    else:
        instantiations = cusr.make_tile_static_eval_instantiations(
            args.kernels,
            ast_capacity=args.ast_capacity,
            tile_rows=args.tile_rows,
            threads_per_cta=args.threads,
        )
        compile_start_ns = time.perf_counter_ns()
        compiled = cusr.compile_tile_static_eval_cubin(instantiations, arch=arch)
        compile_ms = (time.perf_counter_ns() - compile_start_ns) * 1.0e-6
        epilogue = cusr.PatchEpilogue.VALUE

    inspect_start_ns = time.perf_counter_ns()
    inspection = cusr.inspect_cubin(
        compiled.cubin,
        compiled.lowered_names,
        ast_capacity=args.ast_capacity,
        expected_occurrences=1,
    )
    layout = cusr.prepare_patch_layout(inspection)
    inspect_ms = (time.perf_counter_ns() - inspect_start_ns) * 1.0e-6
    patched_cubin = bytearray(compiled.cubin)
    patch_stats, patch_samples = _time_patching(
        layout,
        programs,
        patched_cubin,
        epilogue,
        args.patch_iterations,
    )

    torch.cuda.synchronize(args.device)
    load_start_ns = time.perf_counter_ns()
    if args.family == "mse":
        module = cusr.TileStaticMseModule(patched_cubin, compiled.lowered_names, instantiations, device=args.device)
    else:
        module = cusr.TileStaticEvalModule(patched_cubin, compiled.lowered_names, instantiations, device=args.device)
    torch.cuda.synchronize(args.device)
    module_load_ms = (time.perf_counter_ns() - load_start_ns) * 1.0e-6

    max_stream_count = max(args.streams)
    all_streams = tuple(torch.cuda.Stream(device=args.device) for _ in range(max_stream_count))
    stream_groups = {count: all_streams[:count] for count in args.streams}
    timings = tuple(
        _time_runtime(
            module,
            args.family,
            x_cuda,
            target_cuda,
            masks_cuda,
            words_cuda,
            args.asts_per_kernel,
            output,
            workers,
            args.warmup_iterations,
            args.run_iterations,
        )
        for workers in stream_groups.values()
    )

    check_asts = min(args.check_asts, total_asts)
    check_settings = min(args.check_settings, settings)
    check_rows = min(args.check_rows, rows)
    max_abs_error, max_rel_error = _validate_output(
        args.family,
        programs,
        x,
        target,
        masks,
        words,
        output,
        check_asts,
        check_settings,
        check_rows,
    )

    median_patch_ms = statistics.median(patch_samples)
    best_patch_ms = min(patch_samples)
    median_patch_rate = total_asts / (median_patch_ms * 1.0e-3)
    best_patch_rate = total_asts / (best_patch_ms * 1.0e-3)
    row_evaluations = total_asts * settings * rows

    print(
        f"configuration family={args.family} arch={arch} kernels={args.kernels} "
        f"ast_capacity={args.ast_capacity} asts_per_kernel={args.asts_per_kernel} "
        f"rows={rows} settings={settings} columns={args.columns} ast_mode={args.ast_mode}"
    )
    print(
        f"prepare nvrtc_ms={compile_ms:.3f} inspect_ms={inspect_ms:.3f} "
        f"cubin_mib={len(compiled.cubin) / (1024.0 ** 2):.3f} output_gib={output_gib:.3f}"
    )
    print(
        f"patch iterations={args.patch_iterations} asts={patch_stats.asts_patched} "
        f"median_ms={median_patch_ms:.3f} best_ms={best_patch_ms:.3f} "
        f"median_asts_per_s={median_patch_rate:.3f} best_asts_per_s={best_patch_rate:.3f} "
        f"median_us_per_ast={median_patch_ms * 1000.0 / total_asts:.6f}"
    )
    print(
        f"module_load ms={module_load_ms:.3f} asts_per_s={total_asts / (module_load_ms * 1.0e-3):.3f} "
        f"us_per_ast={module_load_ms * 1000.0 / total_asts:.6f}"
    )
    print("streams host_submit_ms host_us_per_kernel gpu_ms synchronized_ms row_evals_per_s")
    for timing in timings:
        row_rate = row_evaluations / (timing.gpu_elapsed_ms * 1.0e-3)
        print(
            f"{timing.stream_count:7d} {timing.host_submit_ms:14.6f} "
            f"{timing.host_submit_ms * 1000.0 / args.kernels:18.6f} "
            f"{timing.gpu_elapsed_ms:7.3f} {timing.synchronized_ms:15.3f} {row_rate:15.3f}"
        )
    print(
        f"validation asts={check_asts} settings={check_settings} rows={rows if args.family == 'mse' else check_rows} "
        f"max_abs_error={max_abs_error:.9g} max_rel_error={max_rel_error:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
