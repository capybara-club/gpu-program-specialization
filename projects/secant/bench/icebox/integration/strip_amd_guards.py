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
"""Remove branches disabled by SECANT_BENCH_HAS_HIP/HSACO=0."""

from __future__ import annotations

import re
import sys
from pathlib import Path


AMD_MACROS = {
    "SECANT_BENCH_HAS_HIP": 0,
    "SECANT_BENCH_HAS_HSACO": 0,
}


def evaluate(expression: str) -> bool:
    for name, value in AMD_MACROS.items():
        expression = re.sub(rf"\b{name}\b", str(value), expression)
    expression = re.sub(r"\bSECANT_BENCH_PIPELINE\b", "1", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!\s*(?!=)", " not ", expression)
    if re.search(r"[^0-9()\snotandor]", expression):
        raise ValueError(f"cannot evaluate expression: {expression!r}")
    return bool(eval(expression, {"__builtins__": {}}, {}))


def strip(path: Path) -> None:
    lines = path.read_text().splitlines(keepends=True)
    output: list[str] = []
    stack: list[dict[str, bool]] = []

    def enabled() -> bool:
        return all(frame["active"] for frame in stack)

    for line in lines:
        match = re.match(r"^(\s*)#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$", line)
        if match is None:
            if enabled():
                output.append(line)
            continue

        directive = match.group(2)
        argument = match.group(3).strip()
        mentions_amd = any(name in argument for name in AMD_MACROS)

        if directive in {"if", "ifdef", "ifndef"}:
            parent_enabled = enabled()
            managed = mentions_amd
            if managed:
                if directive == "ifdef":
                    condition = argument in AMD_MACROS
                elif directive == "ifndef":
                    condition = argument not in AMD_MACROS
                else:
                    condition = evaluate(argument)
                stack.append(
                    {
                        "managed": True,
                        "parent": parent_enabled,
                        "active": parent_enabled and condition,
                        "taken": condition,
                    }
                )
            else:
                if parent_enabled:
                    output.append(line)
                stack.append(
                    {
                        "managed": False,
                        "parent": parent_enabled,
                        "active": parent_enabled,
                        "taken": False,
                    }
                )
            continue

        if not stack:
            raise ValueError(f"{path}: unmatched #{directive}")
        frame = stack[-1]

        if directive == "elif":
            if frame["managed"]:
                condition = evaluate(argument)
                frame["active"] = (
                    frame["parent"] and not frame["taken"] and condition
                )
                frame["taken"] = frame["taken"] or condition
            elif frame["parent"]:
                output.append(line)
            continue

        if directive == "else":
            if frame["managed"]:
                frame["active"] = frame["parent"] and not frame["taken"]
                frame["taken"] = True
            elif frame["parent"]:
                output.append(line)
            continue

        stack.pop()
        if not frame["managed"] and frame["parent"]:
            output.append(line)

    if stack:
        raise ValueError(f"{path}: unterminated conditional")
    path.write_text("".join(output))


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} FILE...", file=sys.stderr)
        return 1
    for name in sys.argv[1:]:
        strip(Path(name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
