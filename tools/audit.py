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
# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
"""Audit publishable source, licenses, provenance and handoff links.

This is a focused local check, not a guarantee that a secret scanner detects
every possible secret or that third-party legal provenance is infallible.
"""
import ast
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".venv", "node_modules", "target", ".pytest_cache"}
FORBIDDEN = {".cubin", ".fatbin", ".so", ".a", ".o", ".dylib", ".pyc", ".zip", ".pdf", ".npy", ".npz", ".log", ".jsonl", ".parquet", ".rs"}
EXCLUDED_BUILD_FILES = {"Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml"}
SECRETS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\b[0-9]{8,12}:[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{25,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def files():
    if (ROOT / ".git").is_dir():
        data = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
        return sorted(set(ROOT / p for p in data.decode().split("\0") if p))
    answer = []
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(("build", "cmake-build"))]
        answer.extend(Path(directory) / n for n in names)
    return sorted(answer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-snapshot", action="store_true", help="Require owned files to match the initial export, as well as original third-party bytes")
    args = parser.parse_args()
    problems = []
    changed_owned = 0
    manifest = json.loads((ROOT / "docs/source-manifest.json").read_text())
    records = {f["path"]: f for f in manifest["files"]}
    candidates = files()
    total = 0
    for path in candidates:
        rel = path.relative_to(ROOT).as_posix()
        record = records.get(rel, {})
        if path.is_symlink():
            if not path.exists() or not path.resolve().is_relative_to(ROOT):
                problems.append(f"Broken/external symlink: {rel}")
            continue
        if not path.is_file():
            problems.append(f"Missing/non-file source: {rel}")
            continue
        data = path.read_bytes()
        total += len(data)
        if len(data) > 12_000_000:
            problems.append(f"File exceeds source size ceiling: {rel}")
        if path.suffix in FORBIDDEN or path.name in EXCLUDED_BUILD_FILES or path.name.startswith(".env") or any(x in {"scratch", "runs", "logs", "datasets", "worktrees", ".git"} for x in path.relative_to(ROOT).parts):
            problems.append(f"Excluded artifact: {rel}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"Non-text file: {rel}")
            continue
        if "\0" in text:
            problems.append(f"Binary data: {rel}")
        if any(pattern.search(text) for pattern in SECRETS):
            problems.append(f"Possible credential (contents suppressed): {rel}")
        if record.get("license", "MIT") != "MIT":
            if sha(data) != record["source_sha256"]:
                problems.append(f"Third-party file changed from original: {rel}")
        elif not rel.startswith("LICENSES/"):
            notice = text[:4000]
            sidecar = Path(str(path) + ".license")
            if sidecar.exists():
                notice += sidecar.read_text()
            if "Charles Durham" not in notice or not ("SPDX-License-Identifier: MIT" in notice or "MIT License" in notice):
                problems.append(f"Missing owned license notice: {rel}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except ValueError as exc:
                problems.append(f"Invalid JSON: {rel}: {exc}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=rel)
            except SyntaxError as exc:
                problems.append(f"Invalid Python syntax: {rel}:{exc.lineno}: {exc.msg}")
        # New documentation must be self-contained. Historical artifacts are
        # explicitly covered by docs/OMISSIONS.md instead of fabricated links.
        if path.suffix == ".md" and (path.parent == ROOT or path.parent == ROOT / "docs"):
            for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
                link = match.group(1).strip().split("#", 1)[0]
                if not link or "://" in link or link.startswith("mailto:"):
                    continue
                if not (path.parent / unquote(link)).exists():
                    problems.append(f"Broken handoff link: {rel} -> {link}")
    for rel, record in records.items():
        p = ROOT / rel
        if not p.exists():
            problems.append(f"Manifest file absent: {rel}")
        elif not p.is_symlink() and record.get("snapshot_sha256") != sha(p.read_bytes()):
            changed_owned += 1
            if args.verify_snapshot:
                problems.append(f"Snapshot hash changed: {rel}")
    print(f"Audited {len(candidates):,} source/notice files; {total / 1024**2:.2f} MiB.")
    print("Original source licenses:", dict(Counter(f["license"] for f in records.values())))
    if changed_owned:
        print(f"{changed_owned} owned files differ from the initial export; retain their change history in Git.")
    if problems:
        for problem in problems:
            print("FAIL:", problem)
        raise SystemExit(1)
    print("PASS: source size/type, license coverage, third-party byte preservation, JSON/Python syntax, credential patterns, provenance, and handoff links.")


if __name__ == "__main__":
    main()
