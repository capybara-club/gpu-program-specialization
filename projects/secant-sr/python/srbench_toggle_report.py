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
"""Report complete or partial toggle campaigns without dropping failed trials."""
import argparse
import json
from pathlib import Path
import statistics


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def display(value, digits=6):
    return "—" if value is None else f"{value:.{digits}g}"


def report(output):
    campaign = json.loads((output / "campaign.json").read_text())
    manifest = campaign["manifest"]
    config = campaign["identity"]["config"]
    groups = {}
    rows = []
    for job in manifest["jobs"]:
        result_path = output / "trials" / job["id"] / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "pending"}
        groups.setdefault(job["problem"], []).append(result)
        rows.append(result)
    finished = [r for r in rows if r["status"] != "pending"]
    scored = [r for r in rows if r["status"] in {"completed", "no_finite_model"}]
    errors = len(finished) - len(scored)
    lines = [f"# Secant-SR toggle GP: {manifest['suite']}", "",
             f"Host: {campaign.get('hostname', 'unknown')}; GPU slots: {campaign['identity']['gpus']}.",
             f"{len(finished)}/{len(rows)} trials finished; {len(scored)} scored, {errors} errors/timeouts.",
             f"Official split seeds: {manifest['seeds']}; target noise: {manifest['noise']}.",
             "Float32 data/SSE; symbolic assessment not performed. No LM or coefficient refinement.",
             f"Policy: {config['population']:,} ASTs/generation × {config['banks'] * 2 ** config['toggle_bits']:,} "
             f"configurations/AST; up to {config['generations']} generations or {config['seconds']} seconds "
             "(budget checked between generations).", ""]
    if manifest["suite"] == "feynman":
        successes = sum(r.get("accuracy_solution") == 1 for r in scored)
        lines += [f"Numerical threshold: {successes}/{len(scored)} scored trials have test R² > 0.999 "
                  f"({len(rows)} planned). This is not an exact symbolic-recovery count.", ""]
    lines += ["Medians below use available scored trials; pending/error counts remain explicit. "
              "This is not the published median-of-ten aggregate unless that full protocol is separately assessed.", "",
              "| Dataset | Scored/planned | Errors | Numerical successes | Median test R² | Median fit wall (s) | AST occurrences | Configurations |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, trials in sorted(groups.items()):
        valid = [r for r in trials if r["status"] in {"completed", "no_finite_model"}]
        failed = sum(r["status"] not in {"completed", "no_finite_model", "pending"} for r in trials)
        successes = str(sum(r.get("accuracy_solution") == 1 for r in valid)) if manifest["suite"] == "feynman" else "n/a"
        lines.append(f"| {name} | {len(valid)}/{len(trials)} | {failed} | {successes} | "
                     f"{display(median([r.get('validation_r2') for r in valid]))} | "
                     f"{display(median([r.get('process_wall_seconds') for r in valid]), 4)} | "
                     f"{sum(int(r.get('ast_occurrences', 0)) for r in valid):,} | "
                     f"{sum(int(r.get('configurations', 0)) for r in valid):,} |")
    lines += ["", "Counts are evaluated occurrences, including duplicates and unused toggle combinations.",
              "Process wall includes per-fit setup/teardown; summing it across concurrent GPUs is not campaign wall time.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = report(args.campaign)
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
