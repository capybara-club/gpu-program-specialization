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
"""Append curated historical sources from existing local checkouts, without data.

Run after assemble_snapshot.py. This is a maintainer import tool, not a build
dependency. Existing component directories are skipped rather than overwritten.
"""
import argparse
import json
from pathlib import Path
import subprocess
import assemble_snapshot as base

EXTRA = {
    "cuSR": "../git/cuSR",
    "secant-sindy": "../git/secant-sindy",
    "secant-benchmark": "../git/secant-benchmark",
}
TRIALS = ["ast_tools", "cooperative_lm", "cuda_ast_specializer", "rosenbrock_trial", "rosenbrock_cpu_trial", "rfm_dependency_trial", "fitting_batch_trial/plain_cuda_lm"]


def paths(repo):
    return sorted(set(base.git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")) - {""})


def copy_source(repo, source, destination, root, manifest, project):
    path = repo / source
    if not path.is_file() or path.is_symlink():
        return False
    data = path.read_bytes()
    if b"\0" in data or len(data) > 12_000_000:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    out = root / destination
    if out.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    out.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
    license_id = base.foreign_license(source) or "MIT"
    if license_id == "MIT":
        base.license_owned(out)
    manifest["files"].append({"path": destination, "source_project": project, "source_path": source, "source_sha256": base.sha(data), "snapshot_sha256": base.sha(out.read_bytes()), "license": license_id})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Existing code/ directory, with historical git/ as its sibling")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.destination.resolve()
    path = root / "docs/source-manifest.json"
    manifest = json.loads(path.read_text())
    existing = {p["name"] for p in manifest["projects"]}
    for name, relative in EXTRA.items():
        if name in existing:
            continue
        repo = (args.source_root / relative).resolve()
        if (root / "projects" / name).exists():
            raise SystemExit(f"Refusing to overwrite unrecorded component {name}")
        try:
            remote = base.git(repo, "remote", "get-url", "origin").strip()
        except subprocess.CalledProcessError:
            remote = None
        if remote and remote.startswith("git@github.com:"):
            remote = "https://github.com/" + remote.split(":", 1)[1].removesuffix(".git")
        manifest["projects"].append({"name": name, "source_directory": relative, "head": base.git(repo, "rev-parse", "HEAD").strip(), "remote": remote, "dirty": bool(base.git(repo, "status", "--porcelain")), "role": "historical or adjacent research reference; see docs/PROJECTS.md"})
        count = 0
        for source in paths(repo):
            reason = base.excluded(source)
            if reason:
                manifest["omitted"].append({"project": name, "path": source, "reason": reason})
                continue
            count += copy_source(repo, source, f"projects/{name}/{source}", root, manifest, name)
        (root / "projects" / name / "LICENSE").write_text(base.MIT)
        print(name, count)
    repo = args.source_root / "odezza"
    included_trials = set()
    for trial in TRIALS:
        prefix = f"scratch/{trial}/"
        count = 0
        for source in paths(repo):
            if not source.startswith(prefix):
                continue
            relative = source[len(prefix):]
            p = Path(relative)
            # Only top-level source/report files and small named input examples.
            if p.parent != Path(".") and p.parts[0] != "examples":
                continue
            if base.excluded(relative):
                continue
            if p.suffix == ".json" and (p.parent == Path(".") and p.name not in {"pilot.json", "scaling.json"}):
                continue
            destination = f"projects/odezza/research/{trial}/{relative}"
            if copy_source(repo, source, destination, root, manifest, "odezza"):
                count += 1
                included_trials.add(source)
        print(trial, count)
    manifest["omitted"] = [r for r in manifest["omitted"] if not (r["project"] == "odezza" and r["path"] in included_trials)]
    manifest["research_supplement"] = "Selected research sources and their status are described in docs/RESEARCH_COVERAGE.md. Original snapshot hashes remain intact."
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
