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
"""Audit saved version-two service results; this command submits no GPU work."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from examples.search_portfolio.run import feedback


def main():
    rows = []
    folder = HERE / 'runs' / 'sparse-v2'
    for directory in sorted(folder.glob('*-result')):
        name = directory.name[:-7]
        request_raw = (folder / (name + '.json')).read_bytes()
        request = json.loads(request_raw)
        plan = json.loads((folder / (name + '.plan.json')).read_text())
        report_raw = (directory / 'report.json').read_bytes()
        report = json.loads(report_raw)
        submitted = json.loads((directory / 'submitted.json').read_text())
        assert submitted['request_sha256'] == hashlib.sha256(request_raw).hexdigest()
        assert plan['grammar_sha256'] == hashlib.sha256(json.dumps(request['grammar'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert report['status'] == 'complete' and report['retention_complete']
        expected = {f['id']: f for f in plan['families']}
        for family in report['families']:
            assert family['generation_stop'] == 1 and family['pruned'] == 0
            assert family['configuration_limited_derivations'] == 0
            assert family['completed_configurations'] == expected[family['id']]['expected_configurations'], (name, family)
        assert report['counts']['completed_configurations'] == plan['expected_configurations']
        summary = feedback(request, report, plan)
        assert summary['all_families_exhausted']
        for groups in summary['groups'].values():
            assert sum(g['completed_configurations'] for g in groups.values()) == plan['expected_configurations']
        rows.append(dict(name=name,scope=plan.get('validation_scope', 'complete selected profile'),
            families=len(report['families']),counts=report['counts'],timing=report['timing'],
            host_peak_bytes=report['memory']['charged_peak_bytes'],
            generated_variants=sum(f['generated_asts'] for f in report['families']),
            groups=summary['groups'],request_sha256=hashlib.sha256(request_raw).hexdigest(),
            report_sha256=hashlib.sha256(report_raw).hexdigest(),
            underfilled_fraction=sum(d['sizing']['underfilled_configurations'] for d in report['execution']['devices']) / plan['expected_configurations']))
    fitted = json.loads((folder / 'guided-fit.json').read_text())
    next_plan = json.loads((folder / 'guided-next.plan.json').read_text())
    assert next_plan['coefficient_guidance']['source_hashes']['fit'] == hashlib.sha256((folder / 'guided-fit.json').read_bytes()).hexdigest()
    assert not next_plan['coefficient_guidance']['structure_language_changed']
    output = dict(version=2,route='mac1 -> mac3 -> rack1, two RTX 5080s',runs=rows,
        fit_to_followup=dict(passed=True,fit_wall_seconds=fitted['elapsed_seconds'],
            best_training_mse=min(r['best']['mse'] for r in fitted['results'] if r.get('best'))),
        caveats=['Synthetic functional/coverage tests, not blind recovery or an optimum-throughput claim.',
            'Eight-vector smoke and the four-state single-family fixture intentionally underfill GPUs.',
            'Default-profile timing is for four short trajectories (128 RK4 steps/configuration).',
            'Geometric underfill is not measured SM occupancy.'])
    (HERE / 'validation-v2.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(passed=True,requests=len(rows),configurations=sum(r['counts']['completed_configurations'] for r in rows))))


if __name__ == '__main__':
    main()
