#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
"""Maintainer-only snapshot assembler; never imports a source repository's history.

Uses existing local checkouts only. Writes into an empty projects/ directory.
Builds/tests and license auditing are separate: see check_host.py and audit.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SOURCES = {
    "secant": "secant",
    "secant-sr": "secant-sr",
    "odezza": "odezza",
    "secant-system-id": "secant-system-id",
    "mm-ptx": "metamachines/mm-ptx",
    "cubin-function-patch": "metamachines/cubin-function-patch",
    "stack-ptx-emit": "capybara-club/stack-ptx-emit",
    "fused-sindy": "capybara-club/fused-sindy",
    "mm-stack-ptx-ptx-inject-bench": "metamachines/mm-stack-ptx-ptx-inject-bench",
    "mm-stack-ptx-compiler": "metamachines/mm-stack-ptx-compiler",
    "mm-ptx-py": "metamachines/mm-ptx-py",
    "mm-kermac": "metamachines/mm-kermac",
}
EXCLUDE_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    "scratch", "worktrees", "runs", "logs", "datasets", "papers", "target",
    ".vscode", ".idea", "dist", "wheelhouse", "data", "results", "output",
    "outputs", "cache", "caches", "raw", "fortunes-foundation-game",
}
EXCLUDE_SUFFIX = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".zip", ".gz", ".tar",
    ".npy", ".npz", ".parquet", ".arrow", ".db", ".sqlite", ".sqlite3",
    ".so", ".dylib", ".a", ".o", ".obj", ".cubin", ".fatbin", ".pyc",
    ".log", ".jsonl", ".csv", ".tsv", ".svg", ".ptx", ".whl", ".exe",
}
CODE_SUFFIXES = {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".cuh", ".cu", ".js", ".ts", ".css"}
HASH_SUFFIXES = {".py", ".sh", ".bash", ".zsh", ".toml", ".yaml", ".yml", ".cmake", ".mk"}
MIT = """MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
SPDX = "SPDX-FileCopyrightText: 2026 Charles Durham\nSPDX-License-Identifier: MIT\n"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL).decode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def excluded(rel):
    p = Path(rel)
    if p.name == "repository_portfolio_ranked.md":
        return "personal portfolio inventory outside the component documentation"
    if p.suffix == ".rs" or p.name in {"Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml"}:
        return "Rust implementations and build metadata excluded from this handoff"
    if any(x in EXCLUDE_PARTS or x.startswith(("build", "cmake-build")) for x in p.parts):
        # Build helpers are source, unlike build/ output directories.
        if not (len(p.parts) == 1 and p.name in {"build.py", "build.sh"}):
            return "generated output / local workspace / bulk data"
    if p.suffix.lower() in EXCLUDE_SUFFIX:
        return "binary, generated artifact, or raw data"
    if p.name in {".gitmodules", ".DS_Store"} or ".github" in p.parts:
        return "original repository automation or submodule declaration"
    if re.search(r"(?i)(telegram|notify_campaign|secant-notify|secrets|credentials|deploy_|kill_remotes)", rel):
        return "personal operations / notifications"
    if p.name.startswith(".env"):
        return "environment configuration"
    if p.suffix == ".json":
        if ("benchmarks" in p.parts or "bench" in p.parts) and not any(x in p.parts for x in {"examples", "corpus"}) and p.name != "stack_ptx_descriptions.json":
            return "raw benchmark metadata/results (human-readable report retained)"
        if "service_trial/validation/" in rel and "requests/" not in rel and p.name != "million-2048-request.json":
            return "service run output"
        if "examples/" in rel and (p.stem in {"summary", "validation", "validation-v2", "manifest"} or p.stem.endswith(".plan")):
            return "example run output"
    return None


def foreign_license(rel):
    if "boost_subset/" in rel or "/include/boost/" in rel or Path(rel).name == "LICENSE_1_0.txt":
        return "BSL-1.0"
    if Path(rel).name == "incbin.h" or "thirdparty/incbin/" in rel:
        return "Unlicense"
    if Path(rel).name in {"sqlite3.c", "sqlite3.h", "sqlite3ext.h"}:
        return "LicenseRef-SQLite-Public-Domain"
    return None


def license_owned(path):
    """Preserve executable syntax; data/golden files use adjacent REUSE sidecars."""
    text = path.read_text()
    # Covers embedded generator copyright text as well as source headers.
    former_owner = "Charlie" + " Durham"
    text = re.sub(rf"(?i)(copyright[^\n]{{0,80}}?)(Meta\s*Machines(?: LLC)?|{former_owner})", r"\1Charles Durham", text)
    text = re.sub(rf"(SPDX-FileCopyrightText:[^\n]*?)(Meta\s*Machines(?: LLC)?|{former_owner})", r"\1Charles Durham", text, flags=re.I)
    if path.name == "pyproject.toml":
        text = text.replace('name = "MetaMachines LLC", email = "contact@metamachines.co"', 'name = "Charles Durham"')
    if path.name in {"LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"}:
        path.write_text(MIT)
        return
    suffix = path.suffix.lower()
    if suffix in CODE_SUFFIXES:
        header = "/*\n" + "".join(" * " + x + "\n" if x else " *\n" for x in (SPDX + "\n" + MIT).splitlines()) + " */\n"
    elif suffix in HASH_SUFFIXES or path.name in {"Makefile", "CMakeLists.txt", ".gitignore", ".editorconfig", "Dockerfile"} or text.startswith("#!"):
        header = "".join("# " + x + "\n" if x else "#\n" for x in (SPDX + "\n" + MIT).splitlines())
    elif suffix in {".md", ".html"}:
        header = "<!--\n" + SPDX + "\n" + MIT + "-->\n\n"
    else:
        path.write_text(text)
        Path(str(path) + ".license").write_text(SPDX + "\n" + MIT)
        return
    # Existing full MIT headers need only the corrected owner and SPDX marker.
    prefix = text[:2500]
    if "Charles Durham" in prefix and "Permission is hereby granted" in prefix:
        path.write_text(text)
        return
    if text.startswith("#!"):
        first, sep, rest = text.partition("\n")
        text = first + sep + header + rest
    else:
        text = header + text
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_root", type=Path)
    ap.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.destination.resolve()
    if (root / "projects").exists():
        raise SystemExit("Refusing to overwrite existing projects/. Use a fresh destination.")
    (root / "docs").mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "snapshot_date": "2026-09-30", "source_policy": "current working files; no original Git history", "projects": [], "files": [], "omitted": []}
    for name, source in SOURCES.items():
        repo = (args.source_root / source).resolve()
        dest = root / "projects" / name
        try:
            remote = git(repo, "remote", "get-url", "origin").strip()
        except subprocess.CalledProcessError:
            remote = None
        if remote and remote.startswith("git@github.com:"):
            remote = "https://github.com/" + remote.split(":", 1)[1].removesuffix(".git")
        manifest["projects"].append({"name": name, "source_directory": source, "head": git(repo, "rev-parse", "HEAD").strip(), "remote": remote, "dirty": bool(git(repo, "status", "--porcelain"))})
        paths = sorted(set(git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")) - {""})
        for rel in paths:
            src = repo / rel
            reason = excluded(rel)
            if not src.exists() and not src.is_symlink():
                reason = "absent working file"
            elif src.is_dir():
                reason = "external submodule/dependency directory"
            if reason:
                manifest["omitted"].append({"project": name, "path": rel, "reason": reason})
                continue
            out = dest / rel
            if src.is_symlink():
                target = os.readlink(src)
                if Path(target).is_absolute() or not src.resolve().is_relative_to(repo):
                    manifest["omitted"].append({"project": name, "path": rel, "reason": "external symlink"})
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                out.symlink_to(target)
                manifest["files"].append({"path": str(out.relative_to(root)), "source_sha256": sha(target.encode()), "symlink": target, "license": "MIT"})
                continue
            data = src.read_bytes()
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                manifest["omitted"].append({"project": name, "path": rel, "reason": "non-UTF8/binary"})
                continue
            if b"\0" in data:
                manifest["omitted"].append({"project": name, "path": rel, "reason": "binary"})
                continue
            if src.stat().st_size > 12_000_000:
                raise SystemExit(f"Unexpected large source file: {src}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            out.chmod(0o755 if src.stat().st_mode & 0o111 else 0o644)
            foreign = foreign_license(rel)
            if not foreign:
                license_owned(out)
            manifest["files"].append({"path": str(out.relative_to(root)), "source_sha256": sha(data), "snapshot_sha256": sha(out.read_bytes()), "license": foreign or "MIT"})
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "LICENSE").write_text(MIT)
        print(name, len([f for f in manifest["files"] if f["path"].startswith(f"projects/{name}/")]))
    (root / "LICENSE").write_text(MIT)
    (root / "docs" / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "docs" / "source-manifest.json.license").write_text(SPDX + "\n" + MIT)


if __name__ == "__main__":
    main()
