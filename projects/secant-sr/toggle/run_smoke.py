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
"""Small reproducible search panel, not a reliability or SRbench campaign.

Keep every attempt, including failures, with process wall time and exact flags.
No truth expressions are passed to the search. Built-in datasets supply rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    executable = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [root / "secant_sr.h", root / "app/dataset.c"]
    sources += sorted((root / "toggle/src").glob("*.[ch]"))
    sources += sorted((root / "toggle/include").glob("*.h"))
    summary = {
        "scope": "four-case smoke panel; fit threshold is not symbolic equivalence",
        "executable_sha256": digest(executable),
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in sources},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_module_loading": "EAGER",
        "attempts": [],
    }
    env = dict(os.environ, CUDA_MODULE_LOADING="EAGER")
    failed = False
    for repeat in range(args.repeat):
        for problem in ("nguyen1", "nguyen5", "oscillator2", "interaction3"):
            command = [str(executable), "--backend", args.backend, "--problem", problem,
                       "--seed", str(args.seed), "--population", "1024", "--generations", "100",
                       "--seconds", "60", "--rows", "512", "--validation-rows", "512",
                       "--banks", "64", "--constants", "4", "--toggle-bits", "6",
                       "--max-nodes", "31", "--max-depth", "7", "--initial-depth", "3",
                       "--operators", "add,sub,mul,sin,cos", "--elites", "8", "--stop-nmse", "1e-10",
                       "--ast-batch", "256", "--score-mib", "256"]
            if args.backend == "cuda":
                command += ["--pack", "8", "--kernels", "16", "--tile-rows", "128",
                            "--threads", "128", "--workers", "3", "--streams", "8"]
            started = time.perf_counter()
            run = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
            elapsed = time.perf_counter() - started
            stem = args.output / f"{repeat}-{problem}"
            stem.with_suffix(".jsonl").write_text(run.stdout)
            stem.with_suffix(".stderr").write_text(run.stderr)
            result = None
            parse_error = None
            try:
                records = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
                if records and records[-1].get("event") == "result":
                    result = records[-1]
            except ValueError as error:
                parse_error = str(error)
            ok = run.returncode == 0 and result is not None
            failed |= not ok
            attempt = {"problem": problem, "repeat": repeat, "command": command,
                       "process_wall_seconds": elapsed, "exit_code": run.returncode,
                       "parse_error": parse_error, "result": result}
            summary["attempts"].append(attempt)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"problem": problem, "repeat": repeat, "ok": ok,
                              "process_wall_seconds": elapsed,
                              "solved": result.get("solved") if result else None}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
