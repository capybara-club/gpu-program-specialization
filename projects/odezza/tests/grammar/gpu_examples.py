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
"""Execute the handoff populations on small synthetic trajectories, preserving grammar budgets."""
import argparse
import json
import math
from pathlib import Path

from odegrammar.compiler import Compiler, evaluate_postorder
from odegrammar.pools import evaluate_prelude
from odezza.grammar.service import Service, atomic

ROOT = Path(__file__).resolve().parents[2]


def fixture(request):
    # These are execution fixtures, not biological datasets or blind-recovery claims.
    records = Compiler(request).records()
    skeleton = next(r for r in records if r["type"] == "skeleton")
    variant = next(r for r in records if r["type"] == "variant")
    prepared = evaluate_prelude(variant["pools"])
    states = request["states"]
    def rhs(y):
        return [evaluate_postorder(skeleton["rhs"][s], y, constants=prepared["const"], random_values=prepared["rng"],
                    parameters=prepared["param"], toggles=variant["pools"]["toggles"]) for s in states]
    trajectories = []
    for trajectory in range(2):
        times = [0., .02, .04, .07, .1]
        y = [.2+.05*s+.1*trajectory for s in range(len(states))]
        values = [list(y)]
        for a, b in zip(times, times[1:]):
            h = (b-a)/30
            for _ in range(30):
                k1 = rhs(y); k2 = rhs([x+h*.5*k for x, k in zip(y, k1)])
                k3 = rhs([x+h*.5*k for x, k in zip(y, k2)]); k4 = rhs([x+h*k for x, k in zip(y, k3)])
                y = [x+h/6*(a+2*b+2*c+d) for x, a, b, c, d in zip(y, k1, k2, k3, k4)]
            values.append(list(y))
        trajectories.append(dict(times=times, initial=values[0], values=values))
    return dict(states=states, trajectories=trajectories)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--library", required=True); parser.add_argument("--root", required=True)
    parser.add_argument("--cases", nargs="+", default=["adaptation_toggles", "adaptation_search", "gene_circuit", "gene_circuit_constant_grid", "enzyme_pathway"])
    parser.add_argument("--patch-capacity", type=int, default=1024)
    args = parser.parse_args(); service = Service(args.root, library=args.library)
    reports = []
    try:
        for name in args.cases:
            request = json.loads((ROOT/"examples/grammar"/(name+".json")).read_text())
            spec = fixture(request); prepared = service.prepare(spec)
            original_problem_id = request.get("problem_id")
            if "problem_id" in request:
                request["problem_id"] = prepared["problem_id"]
            atomic(Path(args.root)/(name+"-problem.json"), spec)
            job = service.submit(prepared["problem_id"], request, dict(patch_capacity=args.patch_capacity, module_systems=32, max_seconds=1200))
            report = service.jobs[job["job_id"]].result()
            print(name, json.dumps(dict(job_id=job["job_id"], status=report["status"], error=report.get("error"),
                                         generation=report.get("generation"), counts=report["counts"], timing=report["timing"])), flush=True)
            replay_errors = []
            for candidate in report.get("winners", {}).get("global", [])[:3]:
                replay = service.replay(job["job_id"], candidate["id"], cpu=True)
                cpu = replay["cpu_reference"]["mse"]
                error = abs(candidate["mse"]-cpu) if cpu is not None else None
                replay_errors.append(dict(gpu=candidate["mse"], cpu=cpu, absolute_error=error))
                assert error is not None and error < 5e-5*max(1., cpu), replay
            reports.append(dict(name=name, job_id=job["job_id"], status=report["status"], error=report.get("error"),
                                original_problem_id=original_problem_id, effective_problem_id=prepared["problem_id"],
                                generation=report.get("generation"), counts=report["counts"], timing=report["timing"], replay=replay_errors))
            atomic(Path(args.root)/"examples-summary.json", reports)
            assert report["status"] == "complete", report.get("error", report)
    finally: service.close()


if __name__ == "__main__": main()
