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
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


STAGES = (
    "tests",
    "pipeline",
    "native-avx",
    "pysr",
    "operon",
    "kozax",
    "evogp",
    "graph",
)

DEFAULT_ROWS = (
    1024,
    4096,
    16384,
    65536,
    131072,
    262144,
)


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def repo_root():
    return Path(__file__).resolve().parents[1]


def default_build_dir():
    repo = repo_root()
    candidates = (
        repo.parent / "build",
        repo / "build",
    )
    for candidate in candidates:
        if (candidate / "secant_pipeline_bench").is_file():
            return candidate
    return candidates[0]


def default_backend_python():
    candidate = repo_root() / ".baseline-backends" / "bin" / "python"
    return candidate if candidate.is_file() else Path(sys.executable)


def default_pysr_python():
    candidates = (
        repo_root() / ".baseline-pysr" / "bin" / "python",
        repo_root().parent / "pysr_ast_bench" / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_graph_python():
    candidate = repo_root() / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else default_backend_python()


def default_evogp_python():
    candidates = (
        repo_root().parent / "evogp_rocm" / ".venv" / "bin" / "python",
        repo_root().parent / "evogp_trial" / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def default_evogp_repo():
    candidates = (
        repo_root().parent / "evogp_rocm",
        repo_root().parent / "evogp_trial" / "evogp",
    )
    for candidate in candidates:
        if (candidate / "src" / "evogp").is_dir():
            return candidate
    return candidates[-1]


def default_corpus():
    return repo_root() / "bench" / "corpus" / "portable_alu_v1.json"


def command_text(command):
    return " ".join(shlex.quote(str(argument)) for argument in command)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stage_complete(output_dir, name, command, outputs):
    done = output_dir / f".{name}.done.json"
    if not done.is_file():
        return False
    try:
        record = json.loads(done.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = {
        "command": [str(argument) for argument in command],
        "outputs": [str(path) for path in outputs],
    }
    return (
        record.get("command") == expected["command"] and
        record.get("outputs") == expected["outputs"] and
        all(path.is_file() for path in outputs)
    )


def remove_stage_state(output_dir, name, outputs):
    done = output_dir / f".{name}.done.json"
    log = output_dir / f"{name}.log"
    done.unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    for path in outputs:
        path.unlink(missing_ok=True)


def run_stage(
    args,
    name,
    command,
    outputs=(),
    environment=None,
):
    outputs = tuple(outputs)
    if args.force:
        remove_stage_state(args.output_dir, name, outputs)
    if stage_complete(args.output_dir, name, command, outputs):
        print(f"skip stage={name} status=complete", flush=True)
        return
    print(f"stage={name} command={command_text(command)}", flush=True)
    if args.dry_run:
        return

    log_path = args.output_dir / f"{name}.log"
    begin = time.monotonic()
    process_environment = os.environ.copy()
    if environment:
        process_environment.update(environment)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {command_text(command)}\n")
        log.flush()
        process = subprocess.Popen(
            [str(argument) for argument in command],
            cwd=repo_root(),
            env=process_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    seconds = time.monotonic() - begin
    if return_code != 0:
        raise RuntimeError(
            f"stage {name} failed with exit code {return_code}; "
            f"see {log_path}")
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"stage {name} did not produce {missing[0]}")
    write_json(
        args.output_dir / f".{name}.done.json",
        {
            "command": [str(argument) for argument in command],
            "outputs": [str(path) for path in outputs],
            "seconds": seconds,
        },
    )
    print(f"stage={name} status=complete seconds={seconds:.3f}", flush=True)


def require_file(path, description):
    if not path.is_file():
        raise RuntimeError(f"{description} does not exist: {path}")


def check_imports(python, modules, environment=None):
    code = (
        "import importlib\n"
        f"modules={modules!r}\n"
        "for name in modules:\n"
        " importlib.import_module(name)\n"
        "print('imports=' + ','.join(modules))\n"
    )
    process_environment = os.environ.copy()
    if environment:
        process_environment.update(environment)
    process = subprocess.run(
        [str(python), "-c", code],
        text=True,
        capture_output=True,
        env=process_environment,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"dependency preflight failed for {python}: "
            f"{process.stderr.strip()}")


def kozax_container_prefix(args):
    device_group = os.stat("/dev/kfd").st_gid
    repo = repo_root()
    mounts = [repo]
    for path in (args.output_dir, args.corpus.parent):
        if path != repo and repo not in path.parents and path not in mounts:
            mounts.append(path)
    command = [
        args.container_runtime,
        "run",
        "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--group-add", device_group,
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--ipc=host",
        "--env", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
    ]
    for path in mounts:
        command.extend(("--volume", f"{path}:{path}"))
    command.extend([
        "--workdir", repo,
        args.kozax_container_image,
        "python3",
    ])
    return command


def preflight(args):
    enabled = set(args.stages)
    if "pysr" in enabled:
        require_file(args.pysr_python, "PySR sandbox interpreter")
        check_imports(args.pysr_python, ["pysr"])
        julia = (
            args.pysr_python.parent.parent /
            "julia_env" / "pyjuliapkg" / "install" / "bin" / "julia")
        require_file(julia, "PySR sandbox Julia executable")
        project = args.pysr_python.parent.parent / "julia_env"
        process = subprocess.run(
            [
                str(julia),
                f"--project={project}",
                "-e",
                (
                    "using SymbolicRegression; "
                    "using LoopVectorization; "
                    "println(\"julia_backends_ok\")"
                ),
            ],
            text=True,
            capture_output=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "PySR Julia dependency preflight failed: "
                f"{process.stdout.strip()} {process.stderr.strip()}")
    if enabled & {"operon", "kozax"}:
        require_file(args.backend_python, "backend sandbox interpreter")
    if "graph" in enabled:
        require_file(args.graph_python, "graph sandbox interpreter")
    if "operon" in enabled:
        check_imports(args.backend_python, ["numpy", "pyoperon"])
    if "kozax" in enabled:
        code = (
            "import jax, jax.numpy as jnp, kozax\n"
            "devices=jax.devices()\n"
            "print(devices)\n"
            "assert any(device.platform != 'cpu' for device in devices), devices\n"
            "print(jnp.sum(jnp.arange(16, dtype=jnp.float32)).block_until_ready())\n"
        )
        if args.platform == "amd":
            runtime = shutil.which(args.container_runtime)
            if runtime is None:
                raise RuntimeError(
                    f"container runtime was not found: "
                    f"{args.container_runtime}")
            inspect = subprocess.run(
                [
                    runtime,
                    "image",
                    "inspect",
                    args.kozax_container_image,
                ],
                text=True,
                capture_output=True,
            )
            if inspect.returncode != 0:
                detail = inspect.stderr.strip() or inspect.stdout.strip()
                raise RuntimeError(
                    "Kozax ROCm container image inspection failed for "
                    f"{args.kozax_container_image}: {detail}")
            command = [*kozax_container_prefix(args), "-c", code]
            process_environment = None
        else:
            command = [str(args.backend_python), "-c", code]
            process_environment = os.environ.copy()
        process = subprocess.run(
            [str(argument) for argument in command],
            text=True,
            capture_output=True,
            env=process_environment,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Kozax/JAX accelerator preflight failed: "
                f"{process.stdout.strip()} {process.stderr.strip()}")
    if "graph" in enabled:
        check_imports(args.graph_python, ["matplotlib"])
    if "evogp" in enabled:
        require_file(args.evogp_python, "EvoGP sandbox interpreter")
        if not (args.evogp_repo / "src" / "evogp").is_dir():
            raise RuntimeError(
                f"EvoGP source tree does not exist: {args.evogp_repo}")
        code = (
            "import torch\n"
            "assert torch.cuda.is_available()\n"
            "backend = 'rocm' if torch.version.hip is not None else 'cuda'\n"
            "print(backend, torch.cuda.get_device_name())\n"
        )
        process = subprocess.run(
            [str(args.evogp_python), "-c", code],
            text=True,
            capture_output=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"EvoGP accelerator preflight failed: "
                f"{process.stderr.strip()}")


def pipeline_command(args, output):
    command = [
        sys.executable,
        repo_root() / "bench" / "pipeline_sweep.py",
        "--platform", args.platform,
        "--benchmark", args.build_dir / "secant_pipeline_bench",
        "--system", args.system,
        "--output", output,
        "--samples-output",
        output.with_name(output.stem + "_samples.csv"),
        "--shapes", "materialize", "sse",
        "--ast-modes", "alu",
        "--rows", *args.rows,
        "--repeats", "1",
        "--modules", args.pipeline_modules,
        "--workers", args.pipeline_workers,
        "--kernels", args.pipeline_kernels,
        "--asts-per-kernel", args.pipeline_asts_per_kernel,
        "--tile-rows", args.tile_rows,
        "--threads", args.threads,
        "--streams", args.streams,
        "--run-iterations", "1",
        "--source-sm", args.source_sm,
        "--target-sm", args.target_sm,
    ]
    if args.platform == "nvidia":
        command.extend(("--backends", "cuda", "ptx", "cubin"))
        command.extend(("--opt-levels", "0", "1"))
    else:
        command.extend(("--backends", "hip", "hsaco"))
        command.extend(("--opt-levels", "1"))
        command.extend(("--hip-arch", args.hip_arch))
    return command


def baseline_common(args):
    return [
        "--rows", *args.rows,
        "--asts", *args.baseline_asts,
        "--shapes", "materialize", "sse",
        "--warmups", args.baseline_warmups,
        "--min-seconds", args.baseline_min_seconds,
        "--seed", args.seed,
    ]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable, large-scale benchmark campaign for SECANT and "
            "runtime expression backends."))
    parser.add_argument("--platform", choices=("nvidia", "amd"), required=True)
    parser.add_argument("--profile", choices=("smoke", "overnight"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=default_build_dir())
    parser.add_argument("--system")
    parser.add_argument("--rows", type=positive_int, nargs="+")
    parser.add_argument("--baseline-asts", type=positive_int, nargs="+")
    parser.add_argument("--baseline-workers", type=positive_int, nargs="+")
    parser.add_argument("--pipeline-modules", type=positive_int)
    parser.add_argument("--pipeline-workers", type=positive_int)
    parser.add_argument("--pipeline-kernels", type=positive_int)
    parser.add_argument("--pipeline-asts-per-kernel", type=positive_int, default=128)
    parser.add_argument("--tile-rows", type=positive_int, default=8192)
    parser.add_argument("--threads", type=positive_int, default=128)
    parser.add_argument("--streams", type=positive_int, default=8)
    parser.add_argument("--source-sm", type=positive_int, default=80)
    parser.add_argument("--target-sm", type=int)
    parser.add_argument("--hip-arch", default="gfx1201")
    parser.add_argument("--corpus", type=Path, default=default_corpus())
    parser.add_argument("--seed", type=positive_int, default=1)
    parser.add_argument("--baseline-warmups", type=positive_int)
    parser.add_argument("--baseline-repeats", type=positive_int)
    parser.add_argument("--baseline-min-seconds", type=float)
    parser.add_argument("--pysr-python", type=Path, default=default_pysr_python())
    parser.add_argument(
        "--backend-python",
        type=Path,
        default=default_backend_python())
    parser.add_argument(
        "--graph-python",
        type=Path,
        default=default_graph_python())
    parser.add_argument(
        "--evogp-python",
        type=Path,
        default=default_evogp_python())
    parser.add_argument(
        "--evogp-repo",
        type=Path,
        default=default_evogp_repo())
    parser.add_argument("--llvm-path")
    parser.add_argument("--container-runtime", default="docker")
    parser.add_argument(
        "--kozax-container-image",
        default="secant-kozax-rocm:7.1.1")
    parser.add_argument("--stages", choices=STAGES, nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    cpu_count = len(os.sched_getaffinity(0))
    if args.profile == "smoke":
        defaults = {
            "rows": [1024, 4096],
            "baseline_asts": [32, 128],
            "baseline_workers": [1, min(2, cpu_count)],
            "pipeline_modules": 4,
            "pipeline_workers": min(2, cpu_count),
            "pipeline_kernels": 2,
            "baseline_warmups": 1,
            "baseline_repeats": 1,
            "baseline_min_seconds": 0.02,
        }
    else:
        defaults = {
            "rows": list(DEFAULT_ROWS),
            "baseline_asts": [256, 1024, 4096],
            "baseline_workers": [1, cpu_count],
            "pipeline_modules": 240 if args.platform == "nvidia" else 96,
            "pipeline_workers": cpu_count,
            "pipeline_kernels": 64,
            "baseline_warmups": 2,
            "baseline_repeats": 3,
            "baseline_min_seconds": 0.5,
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if args.system is None:
        args.system = (
            "NVIDIA GeForce RTX 5090 + AMD Ryzen 9 9900X"
            if args.platform == "nvidia"
            else "Radeon RX 9070 XT + AMD Ryzen 9 7900X")
    if args.target_sm is None:
        args.target_sm = 0 if args.platform == "nvidia" else 10
    if args.stages is None:
        args.stages = [
            "tests",
            "pipeline",
            "native-avx",
            "pysr",
            "operon",
            "kozax",
            "evogp",
        ]
        args.stages.append("graph")
    if args.baseline_min_seconds <= 0.0:
        parser.error("--baseline-min-seconds must be positive")
    if len(set(args.rows)) != len(args.rows):
        parser.error("--rows contains duplicates")
    if len(set(args.baseline_asts)) != len(args.baseline_asts):
        parser.error("--baseline-asts contains duplicates")
    if len(set(args.baseline_workers)) != len(args.baseline_workers):
        parser.error("--baseline-workers contains duplicates")
    if len(set(args.stages)) != len(args.stages):
        parser.error("--stages contains duplicates")
    if args.streams > args.pipeline_kernels:
        if args.profile == "smoke" and args.streams == 8:
            args.streams = args.pipeline_kernels
        else:
            parser.error("--streams cannot exceed --pipeline-kernels")
    args.output_dir = args.output_dir.resolve()
    args.build_dir = args.build_dir.resolve()
    args.corpus = args.corpus.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main(argv=None):
    args = parse_args(argv)
    print(
        "campaign "
        f"platform={args.platform} profile={args.profile} "
        f"rows={','.join(str(value) for value in args.rows)} "
        f"pipeline_modules={args.pipeline_modules} "
        f"pipeline_workers={args.pipeline_workers} "
        f"kernels_per_module={args.pipeline_kernels} "
        f"asts_per_kernel={args.pipeline_asts_per_kernel} "
        f"total_pipeline_asts="
        f"{args.pipeline_modules * args.pipeline_kernels * args.pipeline_asts_per_kernel}",
        flush=True,
    )

    enabled = set(args.stages)
    if "tests" in enabled:
        run_stage(
            args,
            "tests",
            [
                "cmake",
                "--build", args.build_dir,
                "-j", args.pipeline_workers,
            ],
        )
        run_stage(
            args,
            "ctest",
            [
                "ctest",
                "--test-dir", args.build_dir,
                "--output-on-failure",
            ],
        )
    if not args.dry_run:
        preflight(args)
        require_file(args.corpus, "benchmark corpus")
        if "pipeline" in enabled:
            require_file(
                args.build_dir / "secant_pipeline_bench",
                "SECANT pipeline benchmark")
        if "native-avx" in enabled:
            require_file(
                args.build_dir / "secant_native_avx_bench",
                "native AVX benchmark")
    if args.preflight_only:
        print("preflight status=pass", flush=True)
        return 0

    pipeline_output = args.output_dir / "pipeline.csv"
    pipeline_samples = args.output_dir / "pipeline_samples.csv"
    if "pipeline" in enabled:
        run_stage(
            args,
            "pipeline",
            pipeline_command(args, pipeline_output),
            (pipeline_output, pipeline_samples),
        )

    baseline_outputs = []
    common = baseline_common(args)
    if "native-avx" in enabled:
        output = args.output_dir / "baseline_native_avx.csv"
        baseline_outputs.append(output)
        run_stage(
            args,
            "native-avx",
            [
                sys.executable,
                repo_root() / "bench" / "baselines" / "native_avx_sweep.py",
                "--benchmark", args.build_dir / "secant_native_avx_bench",
                "--output", output,
                *common,
                "--workers", *args.baseline_workers,
                "--min-repeats", args.baseline_repeats,
            ],
            (output,),
        )
    if "pysr" in enabled:
        output = args.output_dir / "baseline_pysr.csv"
        baseline_outputs.append(output)
        run_stage(
            args,
            "pysr",
            [
                args.pysr_python,
                repo_root() / "bench" / "baselines" / "pysr_sweep.py",
                "--output", output,
                "--corpus", args.corpus,
                *common,
                "--workers", *args.baseline_workers,
                "--min-repeats", args.baseline_repeats,
            ],
            (output,),
        )
    if "operon" in enabled:
        output = args.output_dir / "baseline_operon.csv"
        baseline_outputs.append(output)
        run_stage(
            args,
            "operon",
            [
                args.backend_python,
                repo_root() / "bench" / "baselines" / "operon_sweep.py",
                "--output", output,
                "--corpus", args.corpus,
                *common,
                "--workers", *args.baseline_workers,
                "--min-repeats", args.baseline_repeats,
            ],
            (output,),
        )
    if "kozax" in enabled:
        output = args.output_dir / "baseline_kozax.csv"
        baseline_outputs.append(output)
        kozax_asts = sorted(set((64, *args.baseline_asts)))
        environment = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        if args.llvm_path:
            environment["LLVM_PATH"] = args.llvm_path
        command_prefix = (
            kozax_container_prefix(args)
            if args.platform == "amd"
            else [args.backend_python]
        )
        run_stage(
            args,
            "kozax",
            [
                *command_prefix,
                repo_root() / "bench" / "baselines" / "kozax_sweep.py",
                "--output", output,
                "--corpus", args.corpus,
                "--rows", *args.rows,
                "--asts", *kozax_asts,
                "--shapes", "materialize", "sse",
                "--warmups", args.baseline_warmups,
                "--min-repeats", args.baseline_repeats,
                "--min-seconds", args.baseline_min_seconds,
                "--seed", args.seed,
                "--max-estimated-working-bytes",
                12 * 1024 * 1024 * 1024
                if args.platform == "amd"
                else 24 * 1024 * 1024 * 1024,
                "--skip-oversized",
            ],
            (output,),
            environment,
        )
    if "evogp" in enabled:
        output = args.output_dir / "baseline_evogp.csv"
        baseline_outputs.append(output)
        run_stage(
            args,
            "evogp",
            [
                args.evogp_python,
                repo_root() / "bench" / "baselines" / "evogp_sweep.py",
                "--evogp-repo", args.evogp_repo,
                "--output", output,
                "--corpus", args.corpus,
                *common,
                "--min-repeats", args.baseline_repeats,
                "--skip-oversized",
            ],
            (output,),
        )

    if "graph" in enabled:
        if not args.dry_run:
            require_file(pipeline_output, "pipeline CSV")
        if not baseline_outputs:
            baseline_outputs = [
                path for path in (
                    args.output_dir / "baseline_native_avx.csv",
                    args.output_dir / "baseline_pysr.csv",
                    args.output_dir / "baseline_operon.csv",
                    args.output_dir / "baseline_kozax.csv",
                    args.output_dir / "baseline_evogp.csv",
                )
                if path.is_file() or args.dry_run
            ]
        if not args.dry_run:
            for output in baseline_outputs:
                require_file(output, "baseline CSV")
        campaign_csv = args.output_dir / "comparison.csv"
        campaign_svg = args.output_dir / "comparison.svg"
        pipeline_svg = args.output_dir / "pipeline.svg"
        run_stage(
            args,
            "pipeline-graph",
            [
                args.graph_python,
                repo_root() / "bench" / "pipeline_graph.py",
                pipeline_output,
                "--output", pipeline_svg,
            ],
            (pipeline_svg,),
        )
        run_stage(
            args,
            "graph",
            [
                args.graph_python,
                repo_root() / "bench" / "comprehensive_graph.py",
                "--platform", args.platform,
                "--pipeline", pipeline_output,
                "--baselines", *baseline_outputs,
                "--ast-mode", "alu",
                "--output-csv", campaign_csv,
                "--output-svg", campaign_svg,
                "--title", f"Expression Backend Throughput on {args.system}",
            ],
            (campaign_csv, campaign_svg),
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
