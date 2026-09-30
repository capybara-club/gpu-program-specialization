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
"""Command-line front end; all output JSON is strict and streamed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

from .compiler import CompileError, Compiler, SQLiteDedup, compile_lm_request


def read_request(path):
    def reject_constant(value):
        raise CompileError("nonfinite JSON number " + value)
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CompileError("duplicate JSON key " + repr(key))
            result[key] = value
        return result
    if path == "-":
        return json.load(sys.stdin, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Expand ODE grammars into streamed postorder ASTs and execution pool plans; no GPU or solver dependencies.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("compile", "plan", "validate", "lm"):
        sub = subparsers.add_parser(command)
        sub.add_argument("input", help="JSON request path, or - for stdin")
        sub.add_argument("--allow-unsupported-integrator", action="store_true", help="emit annotation-only metadata for future integrators; never substitute rk4")
        if command in ("compile", "lm"):
            sub.add_argument("-o", "--output", default="-", help="output path; default stdout")
        if command in ("compile", "plan"):
            sub.add_argument("--dedup-db", help="fresh SQLite path for disk-backed deduplication")
            sub.add_argument("--max-skeletons", type=int)
            sub.add_argument("--max-variants", type=int)
            sub.add_argument("--max-seconds", type=float)
        if command == "compile":
            sub.add_argument("--compact", action="store_true", help="intern constant banks, RNG descriptors, and numeric plans as shared resource records")
    args = parser.parse_args(argv)
    store = None
    temporary = None
    destination = None
    stream = None
    try:
        request = read_request(args.input)
        if not isinstance(request, dict):
            raise CompileError("request must be a JSON object")
        if args.command in ("compile", "plan"):
            for attribute, key in (("max_skeletons", "max_skeletons"), ("max_variants", "max_variants"), ("max_seconds", "max_seconds")):
                value = getattr(args, attribute)
                if value is not None:
                    limits = request.setdefault("limits", {})
                    if not isinstance(limits, dict):
                        raise CompileError("limits must be an object", path="limits")
                    limits[key] = value
        if args.command == "lm":
            records = iter([compile_lm_request(request, annotation_only=args.allow_unsupported_integrator)])
        else:
            if getattr(args, "dedup_db", None):
                store = SQLiteDedup(args.dedup_db)
            compiler = Compiler(request, annotation_only=args.allow_unsupported_integrator, dedup=store)
            if args.command == "validate":
                print(json.dumps({"valid": True, "format": "odegrammar.postorder.v1", "states": compiler.states, "families": [f["id"] for f in compiler.families], "integration": compiler.integration, "note": "Syntax and declarations checked; no grammar expansion or integration performed."}, allow_nan=False))
                return 0
            records = compiler.records()
            if args.command == "plan":
                for _ in records:
                    pass
                if store is not None:
                    closing, store = store, None
                    closing.close()
                print(json.dumps(compiler.summary, allow_nan=False, sort_keys=True))
                return 0
        if getattr(args, "compact", False):
            from .stream import compact_records
            records = compact_records(records)
        output = getattr(args, "output", "-")
        if output == "-":
            stream = sys.stdout
        else:
            destination = Path(output)
            fd, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=destination.parent)
            stream = os.fdopen(fd, "w", encoding="utf-8")
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        if store is not None:
            closing, store = store, None
            closing.close()
        if destination:
            stream.close()
            stream = None
            os.replace(temporary, destination)
            temporary = None
        return 0
    except (ValueError, TypeError, KeyError, OSError, RecursionError, sqlite3.Error) as error:
        payload = error.as_dict() if isinstance(error, CompileError) else {"type": "error", "code": "INVALID_REQUEST", "message": str(error)}
        print(json.dumps(payload, allow_nan=False), file=sys.stderr)
        return 2
    finally:
        if stream is not None and stream is not sys.stdout:
            stream.close()
        if temporary is not None:
            os.unlink(temporary)
        if store is not None:
            try:
                store.close()
            except sqlite3.Error as error:
                print(json.dumps({"type": "error", "code": "DEDUP_CLOSE_FAILED", "message": str(error)}), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
