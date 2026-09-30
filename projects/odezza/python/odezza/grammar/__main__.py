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
"""Local CLI and stdio MCP entry point for direct grammar jobs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys



def device_list(value):
    try:
        devices=[int(x) for x in value.split(',')]
        if not 1<=len(devices)<=8 or len(set(devices))!=len(devices) or any(x<0 or x>2147483647 for x in devices):
            raise ValueError
        return devices
    except ValueError:
        raise argparse.ArgumentTypeError('Use 1..8 distinct GPU indices separated by commas, e.g. 0,1')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".odezza-grammar")
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    parser.add_argument("--runtime-library", default=os.environ.get("ODEZZA_RUNTIME_LIBRARY"))
    parser.add_argument("--library", default=os.environ.get("ODEZZA_CORE_LIBRARY"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--devices", type=device_list, help="Native GPU group, e.g. --devices 0,1")
    parser.add_argument("--in-process", action="store_true", help="Native benchmark mode without process supervision")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities")
    commands.add_parser("mcp")
    for name in ("run", "plan"):
        p = commands.add_parser(name)
        p.add_argument("request", help="Combined request JSON, or grammar JSON with --problem")
        p.add_argument("--problem")
        p.add_argument("--execution"); p.add_argument("--output")
    p = commands.add_parser("results"); p.add_argument("job_id"); p.add_argument("--format", choices=("full", "compact"), default="full")
    p = commands.add_parser("replay"); p.add_argument("job_id"); p.add_argument("candidate_id", nargs="?")
    p.add_argument("--address", help="JSON file containing a compact candidate's origin")
    p.add_argument("--cpu", action="store_true")
    p = commands.add_parser("fit"); p.add_argument("job_id"); p.add_argument("request"); p.add_argument("--lanes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.backend == "native":
        from .native_service import NativeService, capabilities
        from .native_supervisor import SupervisedNativeService
        Service = NativeService if args.in_process else SupervisedNativeService
        library = args.runtime_library
        if args.command in ("plan", "results", "replay", "fit"):
            parser.error("Native handles live in the MCP process; use its tools, or --backend python for legacy commands")
    else:
        from .service import Service, capabilities
        library = args.library
    if args.command == "capabilities":
        value=capabilities()
        if args.backend=='native':value.update(selected_devices=args.devices or [args.device],process_supervision=not args.in_process)
        print(json.dumps(value, indent=2)); return 0
    if args.devices and args.backend!='native':parser.error('--devices requires the native backend')
    options = dict(library=library, device=args.device)
    if args.backend=='native':options['devices']=args.devices
    service = Service(args.root, **options)
    try:
        if args.command == "mcp":
            from .mcp import serve
            serve(service); return 0
        if args.command in ("run", "plan"):
            request = json.loads(Path(args.request).read_text())
            if args.problem:
                request = dict(problem=json.loads(Path(args.problem).read_text()), grammar=request)
            if args.execution: request["execution"] = json.loads(Path(args.execution).read_text())
            request["plan_only"] = args.command == "plan"
            job = service.submit(**request)
            result = service.wait(job["job_id"]) if hasattr(service,"wait") else service.jobs[job["job_id"]].result()
            result = {**result, "job_id": job["job_id"]}
            if args.output: Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        elif args.command == "results": result = service.results(args.job_id, format=args.format)
        elif args.command == "replay": result = service.replay(args.job_id, args.candidate_id,
            address=json.loads(Path(args.address).read_text()) if args.address else None, cpu=args.cpu)
        else: result = service.fit(args.job_id, json.loads(Path(args.request).read_text()), lanes=args.lanes).result()
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(result.get("status") == "failed")
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
