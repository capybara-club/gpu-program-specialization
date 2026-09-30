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
import json
from pathlib import Path
import subprocess
import sys


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive")
    return parsed


def _identity_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or any(item not in {"unique", "identical"} for item in parsed):
        raise argparse.ArgumentTypeError("identity modes must be unique and/or identical")
    return parsed


def _nonnegative_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("values cannot be negative")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the standalone C99 module lifecycle benchmark."
    )
    parser.add_argument("--runner", required=True)
    parser.add_argument("--fixture", action="append", required=True)
    parser.add_argument("--modules", type=int, default=256)
    parser.add_argument("--warmup-modules", type=int, default=4)
    parser.add_argument("--workers", type=_positive_csv, default=(4,))
    parser.add_argument("--loaded-depths", type=_positive_csv, default=(1, 2, 4, 8, 16))
    parser.add_argument(
        "--stream-counts",
        type=_positive_csv,
        help="Optional stream sweep; by default each loaded depth uses the same stream count.",
    )
    parser.add_argument("--identities", type=_identity_csv, default=("unique", "identical"))
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--wait-clocks", type=_nonnegative_csv, default=(0,))
    parser.add_argument("--poll-us", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.modules <= 0 or arguments.warmup_modules <= 0:
        raise ValueError("module counts must be positive")
    if arguments.blocks <= 0 or not 0 < arguments.threads <= 1024:
        raise ValueError("launch dimensions must be positive")
    if arguments.poll_us < 0 or arguments.device < 0:
        raise ValueError("poll interval and device ordinal cannot be negative")

    runs = []
    for fixture_name in arguments.fixture:
        fixture = Path(fixture_name)
        manifest = json.loads((fixture / "manifest.json").read_text())
        cubin = fixture / "module_lifecycle.cubin"
        for workers in arguments.workers:
            for depth in arguments.loaded_depths:
                stream_counts = arguments.stream_counts or (depth,)
                for streams in stream_counts:
                    for wait_clocks in arguments.wait_clocks:
                        for identity in arguments.identities:
                            command = [
                            arguments.runner,
                            "--cubin",
                            str(cubin),
                            "--kernel",
                            manifest["kernel_name"],
                            "--kernels",
                            str(manifest.get("kernel_count", 1)),
                            "--nonce-offset",
                            str(manifest["nonce_offset"]),
                            "--nonce-magic",
                            manifest["nonce_magic_hex"],
                            "--modules",
                            str(arguments.modules),
                            "--warmup-modules",
                            str(arguments.warmup_modules),
                            "--workers",
                            str(workers),
                            "--streams",
                            str(streams),
                            "--loaded-modules",
                            str(depth),
                            "--blocks",
                            str(arguments.blocks),
                            "--threads",
                            str(arguments.threads),
                            "--wait-clocks",
                            str(wait_clocks),
                            "--poll-us",
                            str(arguments.poll_us),
                            "--device",
                            str(arguments.device),
                            "--identity",
                            identity,
                            ]
                            print(
                                f"fixture={fixture.name} workers={workers} depth={depth} streams={streams} wait_clocks={wait_clocks} identity={identity}",
                                file=sys.stderr,
                                flush=True,
                            )
                            completed = subprocess.run(command, text=True, capture_output=True)
                            if completed.returncode != 0:
                                raise RuntimeError(
                                    f"lifecycle runner failed ({completed.returncode})\n"
                                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                                )
                            result = json.loads(completed.stdout)
                            result["fixture_manifest"] = manifest
                            runs.append(result)

    report = {
        "schema": "secant.module_lifecycle_sweep.v1",
        "scope": "one-context C99 eager load/lookup/launch/event/unload lifecycle with resident input/output buffers",
        "timed_modules_per_run": arguments.modules,
        "warmup_modules_per_run": arguments.warmup_modules,
        "worker_counts": arguments.workers,
        "loaded_module_depths": arguments.loaded_depths,
        "stream_counts": arguments.stream_counts,
        "identity_modes": arguments.identities,
        "wait_clocks_per_kernel": arguments.wait_clocks,
        "poll_microseconds": arguments.poll_us,
        "runs": runs,
        "material_deviations": [
            "Each run creates a fresh process and CUDA context; context creation and fixture compilation are excluded from lifecycle timing.",
            "Unique synthetic modules patch a GPU-verified constant-data nonce but retain identical SASS instruction bytes.",
            "BRKPT fixtures approximate code-heavy modules; global-data fixtures isolate image-size behavior but are not equivalent to System ID code-heavy CUBINs.",
            "The benchmark kernel reads one volatile global gate and returns immediately, so agreement with a production run indicates a module-lifecycle bottleneck only when production GPU execution is shorter than the lifecycle path.",
            "Nonzero wait-clocks use a clock64 busy wait that keeps one warp per launch resident but does not reproduce useful-kernel arithmetic, memory traffic, or register pressure.",
        ],
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
