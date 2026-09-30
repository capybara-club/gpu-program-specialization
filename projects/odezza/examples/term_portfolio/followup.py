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
"""Create a new request with fitted background ranges and unchanged structures.

This consumes explicit request/report/fit provenance. It does not submit work,
alter fixed motifs, or claim that narrowing coefficients replaces exploration.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def narrow(request, plan, report, fit, *, candidate_id=None, half_width=.15):
    if type(half_width) not in (int, float) or not math.isfinite(half_width) or half_width <= 0:
        raise ValueError('half_width must be finite and positive')
    eligible = [r for r in fit['results'] if r.get('best') and
                math.isfinite(r['best']['mse']) and r['best']['mse'] >= 0]
    if candidate_id is not None:
        eligible = [r for r in eligible if r['candidate_id'] == candidate_id]
    if not eligible:
        raise ValueError('no fitted candidate satisfies the selection')
    chosen = min(eligible, key=lambda r: r['best']['mse'])
    source = report['candidates'][chosen['candidate_id']]
    if (chosen['source_origin'] != source['origin'] or chosen['source_values'] != source['values']):
        raise ValueError('fit provenance disagrees with the retained candidate')
    if len(source['slots']) != len(chosen['best']['parameters']):
        raise ValueError('parameter/slot count mismatch')
    parameters = dict(zip((s['name'] for s in source['slots']), chosen['best']['parameters']))
    if len(parameters) != len(source['slots']):
        raise ValueError('duplicate parameter identities require a richer fitting interface')
    allowed = {'rng.linear' + s for s in request['problem']['states']} | {'rng.quadratic'}
    centers = {name[4:]: value for name, value in parameters.items() if name in allowed}
    if not centers or any(type(v) not in (int, float) or not math.isfinite(v) for v in centers.values()):
        raise ValueError('no finite fitted background parameters')
    q, m = deepcopy(request), deepcopy(plan)
    changed = []
    for family in q['grammar']['families']:
        for name, center in centers.items():
            rng = family.get('rng', {}).get(name)
            if rng is None:
                raise ValueError('follow-up requires version-two family-local RNG definitions')
            old = deepcopy(rng['transform'])
            if old['kind'] != 'uniform':
                raise ValueError('this adapter narrows uniform background distributions only')
            rng['transform'] = dict(kind='uniform', low=center-half_width, high=center+half_width)
            changed.append(dict(family=family['id'],parameter=name,previous=old,effective=rng['transform']))
    m['coefficient_guidance'] = dict(candidate_id=chosen['candidate_id'], source_origin=source['origin'],
        fitted_training_mse=chosen['best']['mse'], fit_stopped_with=chosen.get('error'),
        half_width=half_width, centers=centers, changed_ranges=changed, structure_language_changed=False,
        exploration='This is a focused branch. Keep the original broad request available; no exploration is automatically submitted.')
    m['grammar_sha256'] = digest(json.dumps(q['grammar'], sort_keys=True, separators=(',', ':')).encode())
    return q, m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate-id')
    p.add_argument('--half-width', type=float, default=.15)
    args = p.parse_args()
    plan_path = args.output.with_suffix('.plan.json')
    if args.output.exists() or plan_path.exists():
        p.error('output request or plan already exists')
    request_raw, report_raw, fit_raw = (path.read_bytes() for path in (args.request, args.report, args.fit))
    fit = json.loads(fit_raw)
    submitted = json.loads(args.report.with_name('submitted.json').read_text())
    if submitted['request_sha256'] != digest(request_raw):
        raise ValueError('report submission belongs to a different request')
    if fit.get('source_report_sha256') != digest(report_raw):
        raise ValueError('fit must carry the exact report SHA; unbound legacy fit files are not accepted')
    request, plan = narrow(json.loads(request_raw), json.loads(args.request.with_suffix('.plan.json').read_text()),
                          json.loads(report_raw), fit, candidate_id=args.candidate_id, half_width=args.half_width)
    plan['coefficient_guidance']['source_hashes'] = dict(request=digest(request_raw), report=digest(report_raw), fit=digest(fit_raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(request, f, indent=2, allow_nan=False)
    with plan_path.open('x') as f:
        json.dump(plan, f, indent=2, allow_nan=False)
    print(json.dumps(dict(request=str(args.output), candidate=plan['coefficient_guidance']['candidate_id'])))


if __name__ == '__main__':
    main()
